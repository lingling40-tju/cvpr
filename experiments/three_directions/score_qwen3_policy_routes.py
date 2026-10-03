"""Apply the frozen Qwen3 route matcher to preterminal policy views.

Uses natural exact-same-start wrong-goal instructions. Teacher input
contains only available observations, never terminal labels or goals.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from prepare_qwen3_policy_route_manifest import digest
from score_qwen3_route_match import SYSTEM, INTRO, ENDING, logit_margin


def history_images(record: dict, root: Path) -> list[Image.Image]:
    paths = [record["initial_image"]] + [turn["image"] for turn in record["turns"]]
    frames = []
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("policy image escapes replay root")
        with Image.open(path) as source:
            frame = source.convert("RGB")
        frame.thumbnail((448, 448), Image.Resampling.BICUBIC)
        frames.append(frame)
    return frames


def sampled_indices(turn: int) -> list[int]:
    return [math.floor(i * turn / 5 + 0.5) for i in range(6)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-hashes", type=Path, required=True)
    parser.add_argument("--expert-analysis", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.shards != 4:
        raise ValueError("exactly four GPU shards required")
    manifest = json.loads(args.policy_manifest.read_text())
    expert = json.loads(args.expert_analysis.read_text())
    if manifest["schema"] != "qwen3_policy_route_manifest_v1" or \
            manifest["source_sha256"]["group_manifest"] != digest(args.group_manifest) or \
            manifest["source_sha256"]["train_dataset"] != digest(args.train_dataset) or \
            expert["schema"] != "qwen3_route_match_analysis_v1" or \
            not expert["gate"]["passed"] or \
            expert["source_sha256"]["model_hash_file"] != digest(args.model_hashes) or \
            expert["source_sha256"]["model_config"] != digest(args.model / "config.json") or \
            expert["source_sha256"]["tokenizer"] != digest(args.model / "tokenizer.json"):
        raise ValueError("teacher/model/manifest source mismatch")
    selected = manifest["selected"][args.part][args.shard::args.shards]
    if not selected:
        raise ValueError("empty policy shard")
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    if len(episode_by_id) != len(episodes):
        raise ValueError("duplicate train episode ID")
    root = (args.turn_root / args.part).resolve()
    records = {}
    for group in selected:
        eid = group["episode_id"]
        wrong_id = group["wrong_episode_id"]
        if eid not in episode_by_id or wrong_id not in episode_by_id:
            raise ValueError("wrong instruction source unavailable")
        for route in group["routes"]:
            path = root / "records" / f"{route['record_id']}.json"
            if digest(path) != route["record_sha256"]:
                raise ValueError(f"replay hash mismatch {route['record_id']}")
            record = json.loads(path.read_text())
            if record["record_id"] != route["record_id"] or \
                    str(record["episode_id"]) != eid or \
                    record["scene_id"] != group["scene_id"] or \
                    record["instruction"].strip() != \
                    episode_by_id[eid]["instruction"]["instruction_text"].strip() or \
                    len(record["turns"]) != route["turns"] or \
                    (record["terminal_mode"] == "successfully reached the goal.") != \
                    route["success_for_analysis_only"]:
                raise ValueError(f"replay identity mismatch {route['record_id']}")
            records[route["record_id"]] = record
    if hashlib.sha256((SYSTEM + INTRO + ENDING).encode()).hexdigest() != \
            expert["source_sha256"]["prompt"]:
        raise ValueError("teacher prompt changed")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    a_tokens = processor.tokenizer.encode("A", add_special_tokens=False)
    b_tokens = processor.tokenizer.encode("B", add_special_tokens=False)
    if a_tokens != [32] or b_tokens != [33]:
        raise ValueError("A/B tokenizer tokens changed")
    started = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="auto",
        local_files_only=True).eval()
    groups = []
    for group_index, group in enumerate(selected):
        original = episode_by_id[group["episode_id"]]["instruction"]["instruction_text"].strip()
        wrong = episode_by_id[group["wrong_episode_id"]]["instruction"]["instruction_text"].strip()
        output_group = {"seed": group["seed"], "episode_id": group["episode_id"],
                        "scene_id": group["scene_id"],
                        "wrong_episode_id": group["wrong_episode_id"], "routes": []}
        for route in group["routes"]:
            record = records[route["record_id"]]
            source_frames = history_images(record, root)
            try:
                states = []
                for turn in [0] + route["anchors"]:
                    indices = sampled_indices(turn)
                    if max(indices) > len(record["turns"]) or \
                            turn != 0 and len(record["turns"]) <= turn:
                        raise ValueError("future or terminal frame requested")
                    frames = [source_frames[index] for index in indices]
                    first, hash_first, tokens_first = logit_margin(
                        model, processor, frames, original, wrong, 32, 33)
                    swapped, hash_swapped, tokens_swapped = logit_margin(
                        model, processor, frames, wrong, original, 32, 33)
                    states.append({"turn": turn, "sampled_frame_indices": indices,
                                   "correct_margin_first": first,
                                   "correct_margin_swapped": -swapped,
                                   "correct_margin_average": (first - swapped) / 2,
                                   "prompt_sha256": [hash_first, hash_swapped],
                                   "input_tokens": [tokens_first, tokens_swapped]})
                output_group["routes"].append({"record_id": route["record_id"],
                                               "variant": route["variant"],
                                               "terminal_mode_for_analysis_only":
                                                   record["terminal_mode"],
                                               "success_for_analysis_only":
                                                   route["success_for_analysis_only"],
                                               "states": states})
            finally:
                for frame in source_frames:
                    frame.close()
            if args.smoke_only:
                break
        groups.append(output_group)
        if args.smoke_only:
            break
        print(f"{args.part} shard {args.shard} group {group_index + 1}/{len(selected)} "
              f"elapsed_s={time.time() - started:.1f}", flush=True)
        partial = args.output.with_suffix(".partial.json")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(groups, indent=2) + "\n")
    routes_count = sum(len(group["routes"]) for group in groups)
    state_count = sum(len(route["states"]) for group in groups for route in group["routes"])
    payload = {"schema": "qwen3_policy_route_shard_v1", "part": args.part,
               "smoke_only": args.smoke_only, "shard": args.shard, "shards": args.shards,
               "group_count": len(groups), "route_count": routes_count,
               "state_count": state_count, "query_count": 2 * state_count,
               "source_sha256": {"policy_manifest": digest(args.policy_manifest),
                                 "group_manifest": digest(args.group_manifest),
                                 "train_dataset": digest(args.train_dataset),
                                 "model_hash_file": digest(args.model_hashes),
                                 "expert_analysis": digest(args.expert_analysis)},
               "teacher_prompt_sha256": expert["source_sha256"]["prompt"],
               "sample_rule": "six indices floor(i*turn/5+0.5), i=0..5; turn 0 repeats initial view",
               "elapsed_seconds": time.time() - started, "groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, args.output)
    print(json.dumps({"groups": len(groups), "routes": routes_count,
                      "states": state_count, "queries": 2 * state_count,
                      "elapsed_seconds": payload["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
