"""Score a frozen temporal/visual reward on two untouched pair sets.

The 1:1 fusion rule and fit-only scales are read from the prior development
probe. The v2 audit has been used to assess the temporal component before;
the seed-33 pairs are episode-disjoint from fitting but share train scenes.
Neither is val-unseen navigation or independent human annotation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModel, AutoProcessor

from probe_fused_reward import accuracy
from train_siglip_lora_goal_policy import digest, encode_images, encode_texts, score
from train_temporal_progress_encoder import TemporalPotential, predict


def interval(hits: list[bool], scenes: list[str], seed: int = 11) -> list[float]:
    grouped = defaultdict(list)
    for hit, scene in zip(hits, scenes):
        grouped[scene].append(int(hit))
    names = sorted(grouped)
    rng = random.Random(seed)
    values = []
    for _ in range(10000):
        sampled = [value for _ in names
                   for value in grouped[rng.choice(names)]]
        values.append(sum(sampled) / len(sampled))
    values.sort()
    return [values[250], values[9749]]


def collect(name: str, pairs: list[dict], features_path: Path,
            swaps_root: Path, swaps_manifest_sha: str, frames_root: Path,
            manifest_sha: str, encoder, visual, processor,
            scales: dict) -> dict:
    cache = torch.load(features_path, map_location="cpu", weights_only=False)
    if cache["manifest_sha256"] != manifest_sha or \
            cache["hidden"].shape != (len(cache["record_ids"]), 4, 2048) or \
            len(set(cache["record_ids"])) != len(cache["record_ids"]):
        raise ValueError(f"{name} cache provenance mismatch")
    record_indices = {rid: index for index, rid in enumerate(cache["record_ids"])}
    hidden = []
    wrong_hidden = []
    images = []
    instructions = []
    wrong_instructions = []
    for pair in pairs:
        pid = pair["pair_id"]
        for role in ("success", "failure"):
            record_id = pid + "_" + role
            if record_id not in record_indices:
                raise ValueError(f"{name} feature missing {record_id}")
            record = json.loads((frames_root / "records" /
                                 f"{record_id}.json").read_text())
            if record["record_id"] != record_id or len(record["frames"]) != 4:
                raise ValueError(f"{name} RGB record mismatch {record_id}")
            images.append(frames_root / record["frames"][-1]["image"])
            hidden.append(cache["hidden"][record_indices[record_id]])
        swap = torch.load(swaps_root / "records" / f"{pid}.pt",
                          map_location="cpu", weights_only=False)
        if swap["pair_id"] != pid or \
                swap["manifest_sha256"] != swaps_manifest_sha or \
                swap["model_config_sha256"] != cache["model_config_sha256"]:
            raise ValueError(f"{name} swap provenance mismatch {pid}")
        wrong_hidden.append(swap["hidden"])
        instructions.append(pair["instruction"])
        wrong_instructions.append(pair["swapped_instruction"])
    with torch.no_grad():
        temporal = predict(encoder,
                           F.normalize(torch.stack(hidden).float(), dim=-1)
                           .reshape(-1, 2, 4, 2048), torch.device("cuda:0"))
        wrong_temporal = predict(encoder,
                                 F.normalize(torch.stack(wrong_hidden).float(), dim=-1)
                                 .unsqueeze(1), torch.device("cuda:0"))[:, 0]
        image = encode_images(visual, processor, images, batch_size=1).reshape(-1, 2, 768)
        correct = encode_texts(visual, processor, instructions, batch_size=1)
        wrong = encode_texts(visual, processor, wrong_instructions, batch_size=1)
    margins = {
        "temporal": {
            "endpoint": (temporal[:, 0, -1] - temporal[:, 1, -1]).cpu(),
            "grounding": (temporal[:, 0, -1] - wrong_temporal[:, -1]).cpu()},
        "visual": {
            "endpoint": (score(image[:, 0], correct) - score(image[:, 1], correct)).cpu(),
            "grounding": (score(image[:, 0], correct) - score(image[:, 0], wrong)).cpu()}}
    normalized = {source: {task: margin / scales[source][task]
                           for task, margin in tasks.items()}
                  for source, tasks in margins.items()}
    normalized["equal_fusion"] = {
        task: .5 * (normalized["temporal"][task] + normalized["visual"][task])
        for task in ("endpoint", "grounding")}
    results = {}
    scenes = [pair["scene_id"] for pair in pairs]
    for source, tasks in normalized.items():
        results[source] = {}
        for task, margin in tasks.items():
            hits = (margin > 0).tolist()
            results[source][task] = {**accuracy(margin),
                                     "scenes": len(set(scenes)),
                                     "scene_bootstrap95": interval(hits, scenes)}
    per_pair = [{"pair_id": pair["pair_id"], "scene_id": pair["scene_id"],
                 "margins": {source: {task: float(margins[task][i])
                                     for task in ("endpoint", "grounding")}
                             for source, margins in normalized.items()}}
                for i, pair in enumerate(pairs)]
    return {"name": name, "source_manifest_sha256": manifest_sha,
            "summary": results, "per_pair": per_pair}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--v2-features", type=Path, required=True)
    parser.add_argument("--v2-swaps-root", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-frames-root", type=Path, required=True)
    parser.add_argument("--novel-manifest", type=Path, required=True)
    parser.add_argument("--novel-features", type=Path, required=True)
    parser.add_argument("--novel-swaps-manifest", type=Path, required=True)
    parser.add_argument("--novel-swaps-root", type=Path, required=True)
    parser.add_argument("--novel-frames-root", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--siglip-adapter", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    development = json.loads(args.development_report.read_text())
    if development["schema"] != "equal_temporal_visual_reward_development_v3_online_parity" or \
            development["summary"]["equal_fusion"]["development"]["endpoint"]["rate"] < .80 or \
            development["summary"]["equal_fusion"]["development"]["grounding"]["rate"] < .75 or \
            development["policy_manifest_sha256"] != digest(args.policy_manifest) or \
            development["v2_manifest_sha256"] != digest(args.v2_manifest):
        raise ValueError("frozen development gate/provenance failed")
    policy = json.loads(args.policy_manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    novel = json.loads(args.novel_manifest.read_text())
    novel_swaps = json.loads(args.novel_swaps_manifest.read_text())
    if v2["source_manifest_sha256"] != digest(args.policy_manifest) or \
            novel["prior_manifest_sha256"] != digest(args.policy_manifest) or \
            novel_swaps["source_manifest_sha256"] != digest(args.novel_manifest):
        raise ValueError("holdout manifest provenance mismatch")
    v2_rows = {row["pair_id"]: row for row in v2["pairs"]}
    audit = [{**pair,
              "swapped_instruction": v2_rows[pair["pair_id"]]["swapped_instruction"]}
             for pair in policy["pairs"]
             if v2_rows[pair["pair_id"]]["split"] == "audit"]
    novel_rows = {row["pair_id"]: row for row in novel_swaps["pairs"]}
    novel_pairs = [{**pair,
                    "swapped_instruction": novel_rows[pair["pair_id"]]["swapped_instruction"]}
                   for pair in novel["pairs"]]
    if len(audit) != 48 or len(novel_pairs) != 38:
        raise ValueError("holdout pair count mismatch")
    encoder = TemporalPotential().to("cuda:0").eval()
    checkpoint = torch.load(args.v2_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["source_manifest_sha256"] != digest(args.policy_manifest) or \
            checkpoint["v2_manifest_sha256"] != digest(args.v2_manifest):
        raise ValueError("temporal checkpoint provenance mismatch")
    encoder.load_state_dict(checkpoint["model"])
    processor = AutoProcessor.from_pretrained(str(args.siglip_model), local_files_only=True)
    base = AutoModel.from_pretrained(str(args.siglip_model), local_files_only=True,
                                     torch_dtype=torch.bfloat16)
    visual = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"])).cuda().eval()
    adapter = torch.load(args.siglip_adapter, map_location="cpu", weights_only=False)
    if adapter["policy_manifest_sha256"] != digest(args.policy_manifest) or \
            adapter["v2_manifest_sha256"] != digest(args.v2_manifest) or \
            adapter["model_config_sha256"] != digest(args.siglip_model / "config.json") or \
            adapter.get("text_padding") != "fixed_max_length_64":
        raise ValueError("visual checkpoint provenance mismatch")
    set_peft_model_state_dict(visual, adapter["adapter"])
    scales = development["fit_mean_absolute_margin_scales"]
    v2_result = collect("reused_v2_audit", audit, args.v2_features,
                        args.v2_swaps_root, digest(args.v2_manifest),
                        args.v2_frames_root, digest(args.policy_manifest),
                        encoder, visual, processor, scales)
    novel_result = collect("episode_disjoint_seed33", novel_pairs, args.novel_features,
                           args.novel_swaps_root, digest(args.novel_swaps_manifest),
                           args.novel_frames_root, digest(args.novel_manifest),
                           encoder, visual, processor, scales)
    report = {"schema": "equal_temporal_visual_reward_heldout_v3_online_parity",
              "interpretation": "Train-scene offline trajectory probes; no val-unseen navigation or online RL.",
              "development_report_sha256": digest(args.development_report),
              "v2_audit": v2_result, "seed33_novel": novel_result,
              "predeclared_joint_0_75_gate": {
                  subset: {task: result["summary"]["equal_fusion"][task]["rate"] >= .75
                           for task in ("endpoint", "grounding")}
                  for subset, result in (("v2_audit", v2_result),
                                         ("seed33_novel", novel_result))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v2_audit": v2_result["summary"],
                      "seed33_novel": novel_result["summary"],
                      "gate": report["predeclared_joint_0_75_gate"]}, indent=2))


if __name__ == "__main__":
    main()
