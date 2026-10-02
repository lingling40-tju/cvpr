"""Test whether a frozen temporal potential follows the destination words.

For each audit success trajectory, the same four cached RGB frames are
re-encoded with a preselected, same-scene different-goal instruction. The
model must prefer progress under the correct instruction. No simulator goal
coordinate is shown to either model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache_navigation_sft_state import encode_one
from train_temporal_progress_encoder import TemporalPotential


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confidence(hits: list[bool], scenes: list[str]) -> list[float]:
    unique = sorted(set(scenes))
    by_scene = {scene: [hit for hit, name in zip(hits, scenes) if name == scene]
                for scene in unique}
    rng = random.Random(20261003)
    draws = []
    for _ in range(5000):
        selected = [hit for _ in unique for hit in by_scene[rng.choice(unique)]]
        draws.append(sum(selected) / len(selected))
    draws.sort()
    return [draws[125], draws[4875]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--swaps", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    swaps = json.loads(args.swaps.read_text())
    if swaps["source_manifest_sha256"] != digest(args.manifest) or \
            len(swaps["pairs"]) != manifest["counts"]["audit"]:
        raise ValueError("swap/source manifest mismatch")
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    frozen = torch.load(args.encoder, map_location="cpu", weights_only=False)
    if features["manifest_sha256"] != digest(args.manifest) or \
            frozen["manifest_sha256"] != digest(args.manifest) or \
            features["model_config_sha256"] != frozen["feature_model_config_sha256"] or \
            features["prompt_version"] != "vlnce_server_single_observation_v1":
        raise ValueError("representation/encoder mismatch")
    source_index = {pair["pair_id"]: i for i, pair in enumerate(manifest["pairs"])}
    by_pair = {pair["pair_id"]: pair for pair in manifest["pairs"]}
    selected = swaps["pairs"][:args.limit_pairs or None]
    if not selected or args.limit_pairs < 0:
        raise ValueError("empty/invalid swap subset")
    args.output_root.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    token_ids = []
    for word in ("stop", "move", "turn"):
        ids = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"action prefix not one token: {word}")
        token_ids.append(ids[0])
    model_hash = digest(args.model / "config.json")
    if model_hash != features["model_config_sha256"]:
        raise ValueError("SFT checkpoint mismatch")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    encoder = TemporalPotential().cuda().eval()
    encoder.load_state_dict(frozen["model"])
    hits, scenes, margins = [], [], []
    for count, swap in enumerate(selected, 1):
        pair_id = swap["pair_id"]
        pair = by_pair[pair_id]
        if pair["split"] != "audit" or pair["scene_id"] != swap["scene_id"] or \
                pair["episode_id"] != swap["source_episode_id"]:
            raise ValueError(f"invalid audit swap {pair_id}")
        index = source_index[pair_id]
        record_id = pair_id + "_success"
        record = json.loads((args.records_root / "records" /
                             f"{record_id}.json").read_text())
        if record["record_id"] != record_id or len(record["frames"]) != 4 or \
                features["record_ids"][2 * index] != record_id:
            raise ValueError(f"frame/feature mismatch {record_id}")
        out = args.output_root / "records" / f"{pair_id}.json"
        out.parent.mkdir(exist_ok=True)
        if out.is_file():
            result = json.loads(out.read_text())
            if result["swap_manifest_sha256"] != digest(args.swaps) or \
                    result["encoder_sha256"] != digest(args.encoder):
                raise ValueError("stale swap cache")
        else:
            vectors = []
            for frame in record["frames"]:
                vector, _ = encode_one(
                    model, processor, args.records_root / frame["image"],
                    swap["swapped_instruction"], tuple(token_ids),
                    initial=frame["action_index"] == 0)
                vectors.append(vector)
            wrong = torch.stack(vectors).float().cuda().unsqueeze(0)
            correct = features["hidden"][2 * index].float().cuda().unsqueeze(0)
            with torch.inference_mode():
                correct_progress = float(encoder(F.normalize(correct, dim=-1))[0, -1])
                swapped_progress = float(encoder(F.normalize(wrong, dim=-1))[0, -1])
            result = {"pair_id": pair_id, "scene_id": pair["scene_id"],
                      "correct_progress": correct_progress,
                      "swapped_progress": swapped_progress,
                      "correct_over_swapped": correct_progress > swapped_progress,
                      "swap_manifest_sha256": digest(args.swaps),
                      "encoder_sha256": digest(args.encoder),
                      "feature_prompt_version": features["prompt_version"]}
            temp = out.with_suffix(".tmp")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temp, out)
        hits.append(bool(result["correct_over_swapped"]))
        scenes.append(pair["scene_id"])
        margins.append(result["correct_progress"] - result["swapped_progress"])
        if count % 10 == 0 or count == len(selected):
            print(f"counterfactual {count}/{len(selected)}", flush=True)
    report = {"schema": "temporal_instruction_swap_audit_v1",
              "pairs": len(selected), "scenes": len(set(scenes)),
              "correct_instruction_preference": sum(hits) / len(hits),
              "scene_bootstrap95": confidence(hits, scenes),
              "mean_correct_minus_swapped_progress": sum(margins) / len(margins),
              "predeclared_gate": {"correct_instruction_at_least_0_75":
                                       sum(hits) / len(hits) >= .75,
                                   "positive_mean_margin": sum(margins) > 0},
              "swap_manifest_sha256": digest(args.swaps),
              "encoder_sha256": digest(args.encoder),
              "interpretation": "Train-scene instruction counterfactual; not an online RL or val-unseen result."}
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
