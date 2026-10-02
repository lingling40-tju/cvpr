"""Probe a fixed, equal-weight temporal/visual reward on fit and dev scenes.

The temporal encoder reads frozen navigation-SFT states. The visual encoder
reads the same replay RGB and instructions. Each margin is scaled only by
the fit-scene mean absolute margin; the 1:1 fusion weight is not tuned on
development labels. Audit scenes and the seed-33 novel episodes are not read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModel, AutoProcessor

from train_siglip_lora_goal_policy import digest, encode_images, encode_texts, score
from train_temporal_progress_encoder import TemporalPotential, predict


def accuracy(margins: torch.Tensor) -> dict:
    return {"hits": int((margins > 0).sum()), "pairs": margins.numel(),
            "rate": float((margins > 0).float().mean()),
            "mean_margin": float(margins.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--sft-features", type=Path, required=True)
    parser.add_argument("--v2-swaps-root", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-frames-root", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--siglip-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    policy = json.loads(args.policy_manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    if v2["source_manifest_sha256"] != digest(args.policy_manifest) or v2["group_size"] != 4:
        raise ValueError("manifest provenance mismatch")
    rows = {row["pair_id"]: row for row in v2["pairs"]}
    selected = [(index, pair) for index, pair in enumerate(policy["pairs"])
                if rows[pair["pair_id"]]["split"] in ("fit", "development")]
    if len(selected) != v2["counts"]["fit"] + v2["counts"]["development"]:
        raise ValueError("fit/development coverage mismatch")
    features = torch.load(args.sft_features, map_location="cpu", weights_only=False)
    if features["manifest_sha256"] != digest(args.policy_manifest) or \
            features["hidden"].shape != (800, 4, 2048):
        raise ValueError("SFT feature provenance mismatch")
    checkpoint = torch.load(args.v2_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["source_manifest_sha256"] != digest(args.policy_manifest) or \
            checkpoint["v2_manifest_sha256"] != digest(args.v2_manifest):
        raise ValueError("temporal encoder provenance mismatch")
    model = TemporalPotential().to("cuda:0")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    pairs, hidden, wrong_hidden = [], [], []
    visual_paths, correct_texts, wrong_texts = [], [], []
    for index, pair in selected:
        pid = pair["pair_id"]
        for role_index, role in enumerate(("success", "failure")):
            record_id = pid + "_" + role
            if features["record_ids"][2 * index + role_index] != record_id:
                raise ValueError(f"feature order mismatch {record_id}")
            record = json.loads((args.policy_frames_root / "records" /
                                 f"{record_id}.json").read_text())
            if record["record_id"] != record_id or len(record["frames"]) != 4:
                raise ValueError(f"RGB record mismatch {record_id}")
            visual_paths.append(args.policy_frames_root / record["frames"][-1]["image"])
            hidden.append(features["hidden"][2 * index + role_index])
        swap = torch.load(args.v2_swaps_root / "records" / f"{pid}.pt",
                          map_location="cpu", weights_only=False)
        if swap["pair_id"] != pid or \
                swap["manifest_sha256"] != digest(args.v2_manifest) or \
                swap["model_config_sha256"] != features["model_config_sha256"]:
            raise ValueError(f"wrong-instruction feature mismatch {pid}")
        wrong_hidden.append(swap["hidden"])
        correct_texts.append(pair["instruction"])
        wrong_texts.append(rows[pid]["swapped_instruction"])
        pairs.append({"pair_id": pid, "scene_id": pair["scene_id"],
                      "split": rows[pid]["split"]})
    with torch.no_grad():
        temporal = predict(model, F.normalize(torch.stack(hidden).float(), dim=-1)
                           .reshape(-1, 2, 4, 2048), torch.device("cuda:0"))
        wrong_temporal = predict(model,
                                 F.normalize(torch.stack(wrong_hidden).float(), dim=-1)
                                 .unsqueeze(1), torch.device("cuda:0"))[:, 0]
    temporal_endpoint = temporal[:, 0, -1] - temporal[:, 1, -1]
    temporal_grounding = temporal[:, 0, -1] - wrong_temporal[:, -1]
    del model
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(str(args.siglip_model), local_files_only=True)
    base = AutoModel.from_pretrained(str(args.siglip_model), local_files_only=True,
                                     torch_dtype=torch.bfloat16)
    visual = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"])).cuda().eval()
    adapter = torch.load(args.siglip_adapter, map_location="cpu", weights_only=False)
    if adapter["policy_manifest_sha256"] != digest(args.policy_manifest) or \
            adapter["v2_manifest_sha256"] != digest(args.v2_manifest) or \
            adapter["model_config_sha256"] != digest(args.siglip_model / "config.json"):
        raise ValueError("SigLIP adapter provenance mismatch")
    set_peft_model_state_dict(visual, adapter["adapter"])
    with torch.no_grad():
        image = encode_images(visual, processor, visual_paths).reshape(-1, 2, 768)
        correct = encode_texts(visual, processor, correct_texts)
        wrong = encode_texts(visual, processor, wrong_texts)
    visual_endpoint = score(image[:, 0], correct) - score(image[:, 1], correct)
    visual_grounding = score(image[:, 0], correct) - score(image[:, 0], wrong)
    raw = {"temporal": {"endpoint": temporal_endpoint.cpu(),
                        "grounding": temporal_grounding.cpu()},
           "visual": {"endpoint": visual_endpoint.cpu(),
                      "grounding": visual_grounding.cpu()}}
    fit = torch.tensor([row["split"] == "fit" for row in pairs], dtype=torch.bool)
    dev = ~fit
    calibrated = {}
    scales = {}
    for name in raw:
        calibrated[name] = {}
        scales[name] = {}
        for task in ("endpoint", "grounding"):
            scale = float(raw[name][task][fit].abs().mean())
            if scale <= 1e-8:
                raise ValueError(f"zero fit margin scale {name}/{task}")
            scales[name][task] = scale
            calibrated[name][task] = raw[name][task] / scale
    calibrated["equal_fusion"] = {
        task: .5 * (calibrated["temporal"][task] + calibrated["visual"][task])
        for task in ("endpoint", "grounding")}
    summary = {name: {split: {task: accuracy(margin[mask])
                              for task, margin in scores.items()}
                      for split, mask in (("fit", fit), ("development", dev))}
               for name, scores in calibrated.items()}
    report = {"schema": "equal_temporal_visual_reward_development_v1",
              "interpretation": "Exploratory fit/development probe; fixed 1:1 weight; no audit or novel pairs; no RL.",
              "policy_manifest_sha256": digest(args.policy_manifest),
              "v2_manifest_sha256": digest(args.v2_manifest),
              "sft_model_config_sha256": features["model_config_sha256"],
              "siglip_model_config_sha256": adapter["model_config_sha256"],
              "siglip_selected_step": adapter["selected_step"],
              "counts": {"fit": int(fit.sum()), "development": int(dev.sum())},
              "fit_mean_absolute_margin_scales": scales,
              "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
