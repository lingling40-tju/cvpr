"""Screen fixed temporal/visual fusion on unsuccessful trajectory pairs.

This fallback runs only after the failure-only online 256-episode screen has
failed. It reuses frozen SFT features, replayed RGB, and a frozen SigLIP
adapter. Fit scenes set the visual scale; development scenes decide whether
to open the separate 106-pair scene audit. No policy update occurs here.
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


def stats(margin: torch.Tensor) -> dict:
    return {"pairs": margin.numel(), "hits": int((margin > 0).sum()),
            "rate": float((margin > 0).float().mean()),
            "mean_margin": float(margin.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--siglip-adapter", type=Path, required=True)
    parser.add_argument("--old-pair-manifest", type=Path, required=True)
    parser.add_argument("--old-v2-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = json.loads(args.manifest.read_text())
    pairs = manifest["pairs"]
    manifest_sha = digest(args.manifest)
    replay = json.loads(args.replay_summary.read_text())
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    calibration = json.loads(args.calibration.read_text())
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    adapter = torch.load(args.siglip_adapter, map_location="cpu", weights_only=False)
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    if manifest["schema"] != "failure_rank_train_scene_v1" or \
            replay["manifest_sha256"] != manifest_sha or \
            replay["completed_trajectories"] != 1856 or replay["errors"] or \
            cache["manifest_sha256"] != manifest_sha or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048) or \
            checkpoint["manifest_sha256"] != manifest_sha or \
            calibration["new_encoder_sha256"] != digest(args.encoder) or \
            old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            adapter["policy_manifest_sha256"] != digest(args.old_pair_manifest) or \
            adapter["v2_manifest_sha256"] != digest(args.old_v2_manifest) or \
            adapter["model_config_sha256"] != digest(args.siglip_model / "config.json") or \
            adapter.get("text_padding") != "fixed_max_length_64" or \
            adapter["selected_step"] != 768:
        raise ValueError("visual fallback provenance mismatch")
    split = {name: [i for i, pair in enumerate(pairs) if pair["split"] == name]
             for name in ("fit", "development", "audit")}
    if {name: len(indices) for name, indices in split.items()} != \
            {"fit": 697, "development": 125, "audit": 106}:
        raise ValueError("scene split coverage mismatch")
    if any(set(pairs[i]["scene_id"] for i in split[a]) &
           set(pairs[j]["scene_id"] for j in split[b])
           for a, b in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit"))):
        raise ValueError("scene leakage")
    device = torch.device("cuda:0")
    temporal = TemporalPotential().to(device).eval()
    temporal.load_state_dict(checkpoint["model"])
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    processor = AutoProcessor.from_pretrained(str(args.siglip_model),
                                              local_files_only=True)
    base = AutoModel.from_pretrained(str(args.siglip_model),
                                     local_files_only=True,
                                     torch_dtype=torch.bfloat16)
    visual = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"])).cuda().eval()
    set_peft_model_state_dict(visual, adapter["adapter"])

    def collect(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        paths = []
        texts = []
        for index in indices:
            pair = pairs[index]
            texts.append(pair["instruction"])
            for role_index, role in enumerate(("near", "far")):
                record_id = pair["pair_id"] + "_" + role
                if cache["record_ids"][2 * index + role_index] != record_id:
                    raise ValueError("SFT feature identity mismatch")
                record = json.loads((args.replay_root / "records" /
                                     f"{record_id}.json").read_text())
                if record["record_id"] != record_id or \
                        record["manifest_sha256"] != manifest_sha or \
                        len(record["frames"]) != 4:
                    raise ValueError("replayed RGB identity mismatch")
                paths.append(args.replay_root / record["frames"][-1]["image"])
        with torch.inference_mode():
            visual_images = encode_images(visual, processor, paths,
                                          batch_size=1).reshape(-1, 2, 768)
            visual_text = encode_texts(visual, processor, texts,
                                       batch_size=1)
            visual_margin = score(visual_images[:, 0], visual_text) - \
                score(visual_images[:, 1], visual_text)
            temporal_pred = predict(temporal, hidden[indices], device)
            temporal_margin = temporal_pred[:, 0, -1] - temporal_pred[:, 1, -1]
        return temporal_margin.cpu(), visual_margin.cpu()

    fit_t, fit_v = collect(split["fit"])
    dev_t, dev_v = collect(split["development"])
    t_scale = float(calibration["temporal_scale"])
    v_scale = float(fit_v.abs().mean())
    if not .01 < t_scale < 100 or not .01 < v_scale < 100:
        raise ValueError("invalid fit-only scales")
    def summary(t: torch.Tensor, v: torch.Tensor) -> dict:
        return {"temporal": stats(t / t_scale),
                "visual": stats(v / v_scale),
                "equal_fusion": stats(.5 * (t / t_scale + v / v_scale))}
    fit_summary = summary(fit_t, fit_v)
    dev_summary = summary(dev_t, dev_v)
    gate = {"development_fusion_at_least_0_70":
                dev_summary["equal_fusion"]["rate"] >= .70,
            "development_fusion_gain_over_temporal_at_least_5pp":
                dev_summary["equal_fusion"]["rate"] -
                dev_summary["temporal"]["rate"] >= .05 - 1e-8}
    audit_summary = None
    if all(gate.values()):
        audit_t, audit_v = collect(split["audit"])
        audit_summary = summary(audit_t, audit_v)
    result = {"schema": "failure_only_temporal_visual_train_scene_probe_v1",
              "interpretation": "Exploratory train-scene failure-pair probe. The temporal audit was previously inspected. No online RL or val-unseen navigation result.",
              "manifest_sha256": manifest_sha,
              "replay_summary_sha256": digest(args.replay_summary),
              "encoder_sha256": digest(args.encoder),
              "calibration_sha256": digest(args.calibration),
              "siglip_adapter_sha256": digest(args.siglip_adapter),
              "scales_from_fit_only": {"temporal": t_scale, "visual": v_scale},
              "fit": fit_summary, "development": dev_summary,
              "predeclared_development_gate": gate,
              "audit_opened": audit_summary is not None,
              "audit": audit_summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
