"""Audit three paired full val-unseen decodes and the unseen screen complement."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from analyze_direction_eval import summarize
from analyze_matched_pair import compare, load_label


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc"
SEEDS = (11, 22, 33)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(rows_by_seed: dict, ids: list[str], scenes: list[str], metric: str):
    scene_to_ids = defaultdict(list)
    for eid, scene in zip(ids, scenes):
        scene_to_ids[scene].append(eid)
    scene_names = sorted(scene_to_ids)
    sums = {}
    for seed in SEEDS:
        candidate, control = rows_by_seed[seed]
        sums[seed] = {}
        for scene, members in scene_to_ids.items():
            if metric == "sr_pp":
                value = sum(int(bool(candidate[eid]["success"])) -
                            int(bool(control[eid]["success"]))
                            for eid in members)
            else:
                value = sum(float(candidate[eid]["spl"]) -
                            float(control[eid]["spl"])
                            for eid in members)
            sums[seed][scene] = value
    rng = random.Random(20261004 if metric == "sr_pp" else 20261005)
    boots = []
    for _ in range(2000):
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        sampled_scenes = [rng.choice(scene_names) for _ in scene_names]
        denominator = len(sampled_seeds) * sum(
            len(scene_to_ids[scene]) for scene in sampled_scenes)
        numerator = sum(sums[seed][scene] for seed in sampled_seeds
                        for scene in sampled_scenes)
        boots.append(100 * numerator / denominator)
    boots.sort()
    return [boots[49], boots[1949]]


def summarize_scope(rows_by_seed: dict, ids: list[str], scenes: list[str]):
    pairs = []
    for seed in SEEDS:
        candidate, control = rows_by_seed[seed]
        pairs.append({"seed": seed,
                      "candidate_metrics": summarize(candidate, ids),
                      "control_metrics": summarize(control, ids),
                      "paired": compare(candidate, control, ids, scenes)})
    for item in pairs:
        if item["candidate_metrics"]["count"] != len(ids) or \
                item["control_metrics"]["count"] != len(ids) or \
                item["candidate_metrics"]["inference_errors"] != 0 or \
                item["control_metrics"]["inference_errors"] != 0:
            raise ValueError("scope coverage or inference error")
    metrics = {}
    for metric in ("sr_pp", "spl_pp"):
        values = [item["paired"][metric] for item in pairs]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite paired metric")
        metrics[metric] = {
            "per_seed": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "scene_seed_bootstrap95": interval(rows_by_seed, ids, scenes, metric)}
    return {"episodes_per_seed": len(ids), "scenes": len(set(scenes)),
            "pairs": pairs, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full_path = args.root / "manifest.json"
    if digest(full_path) != FULL_SHA or \
            digest(args.screen_manifest) != SCREEN_SHA:
        raise ValueError("fixed val-unseen manifest hash mismatch")
    full = json.loads(full_path.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = [str(item) for item in full["episode_ids"]]
    scenes = [str(item) for item in full["scene_ids"]]
    screen_ids = {str(item) for item in screen["episode_ids"]}
    if len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != 1839 or \
            len(screen_ids) != 256 or not screen_ids.issubset(ids):
        raise ValueError("full/screen episode coverage mismatch")
    outside_ids = [eid for eid in ids if eid not in screen_ids]
    outside_scenes = [scene for eid, scene in zip(ids, scenes)
                      if eid not in screen_ids]
    if len(outside_ids) != 1583:
        raise ValueError("screen complement mismatch")
    rows_by_seed = {}
    pair_hashes = {}
    for seed in SEEDS:
        candidate = f"qwen_group_rank_scale_128_seed{seed}"
        control = f"qwen_exact_scale_control_128_seed{seed}"
        pair_path = args.root / f"paired_{candidate}_vs_{control}.json"
        pair = json.loads(pair_path.read_text())
        if pair["split"] != "val_unseen" or pair["episodes"] != 1839 or \
                pair["manifest_sha256"] != FULL_SHA or \
                pair["candidate"] != candidate or pair["control"] != control or \
                pair["candidate_metrics"]["inference_errors"] != 0 or \
                pair["control_metrics"]["inference_errors"] != 0:
            raise ValueError(f"invalid full paired result seed {seed}")
        rows_by_seed[seed] = (load_label(args.root, candidate, ids, 4),
                              load_label(args.root, control, ids, 4))
        pair_hashes[str(seed)] = digest(pair_path)
    report = {"schema": "qwen_group_rank_full_val_unseen_v1",
              "full_manifest_sha256": FULL_SHA,
              "screen_manifest_sha256": SCREEN_SHA,
              "paired_json_sha256": pair_hashes,
              "seeds": list(SEEDS),
              "full": summarize_scope(rows_by_seed, ids, scenes),
              "outside_reused_screen": summarize_scope(
                  rows_by_seed, outside_ids, outside_scenes),
              "interpretation": "Three paired seeds, one decode per checkpoint/episode. Scene-and-seed bootstrap is exploratory; no independent test split or human semantic truth."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({scope: report[scope]["metrics"]
                      for scope in ("full", "outside_reused_screen")}, indent=2))


if __name__ == "__main__":
    main()
