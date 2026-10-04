"""Audit three paired n=4 oracle/control full-val seeds and screen complement."""

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
SCREEN_SHA = "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c"
SEEDS = (11, 22, 33)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(rows: dict, ids: list[str], scenes: list[str], metric: str) -> list[float]:
    by_scene = defaultdict(list)
    for eid, scene in zip(ids, scenes):
        by_scene[scene].append(eid)
    names = sorted(by_scene)
    sums = {}
    for seed in SEEDS:
        candidate, control = rows[seed]
        sums[seed] = {}
        for scene, members in by_scene.items():
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
    boot = []
    for _ in range(5000):
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        sampled_scenes = [rng.choice(names) for _ in names]
        denominator = len(sampled_seeds) * sum(len(by_scene[s]) for s in sampled_scenes)
        numerator = sum(sums[seed][scene] for seed in sampled_seeds
                        for scene in sampled_scenes)
        boot.append(100 * numerator / denominator)
    boot.sort()
    return [boot[124], boot[4874]]


def scope(rows: dict, ids: list[str], scenes: list[str]) -> dict:
    pairs = []
    for seed in SEEDS:
        candidate, control = rows[seed]
        cm = summarize(candidate, ids)
        bm = summarize(control, ids)
        if cm["count"] != len(ids) or bm["count"] != len(ids) or \
                cm["inference_errors"] or bm["inference_errors"]:
            raise ValueError(f"coverage or inference error seed {seed}")
        pairs.append({"seed": seed, "candidate_metrics": cm,
                      "control_metrics": bm,
                      "paired": compare(candidate, control, ids, scenes)})
    metrics = {}
    for name in ("sr_pp", "spl_pp"):
        values = [pair["paired"][name] for pair in pairs]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite paired metric")
        metrics[name] = {"per_seed": values, "mean": statistics.mean(values),
                         "sample_sd": statistics.stdev(values),
                         "scene_seed_bootstrap95": interval(rows, ids, scenes, name)}
    return {"episodes_per_seed": len(ids), "scenes": len(set(scenes)),
            "pairs": pairs, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full_path = args.root / "manifest.json"
    if digest(full_path) != FULL_SHA or digest(args.screen_manifest) != SCREEN_SHA:
        raise ValueError("frozen val-unseen manifest checksum mismatch")
    full = json.loads(full_path.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = [str(value) for value in full["episode_ids"]]
    scenes = [str(value) for value in full["scene_ids"]]
    screen_ids = {str(value) for value in screen["episode_ids"]}
    if len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != 1839 or \
            len(set(scenes)) != 11 or len(screen_ids) != 256 or \
            not screen_ids.issubset(ids):
        raise ValueError("full or screen episode coverage mismatch")
    outside_ids = [eid for eid in ids if eid not in screen_ids]
    outside_scenes = [scene for eid, scene in zip(ids, scenes)
                      if eid not in screen_ids]
    if len(outside_ids) != 1583:
        raise ValueError("screen complement mismatch")
    rows = {}
    pair_hashes = {}
    for seed in SEEDS:
        candidate = f"oracle_turnwise_exact512_128_seed{seed}"
        control = f"oracle_exact512_control_128_seed{seed}"
        pair_path = args.root / f"paired_{candidate}_vs_{control}.json"
        pair = json.loads(pair_path.read_text())
        if pair["split"] != "val_unseen" or pair["episodes"] != 1839 or \
                pair["manifest_sha256"] != FULL_SHA or \
                pair["candidate"] != candidate or pair["control"] != control or \
                pair["candidate_metrics"]["inference_errors"] or \
                pair["control_metrics"]["inference_errors"]:
            raise ValueError(f"invalid paired report seed {seed}")
        rows[seed] = (load_label(args.root, candidate, ids, 4),
                      load_label(args.root, control, ids, 4))
        pair_hashes[str(seed)] = digest(pair_path)
    report = {
        "schema": "oracle_exact512_three_seed_full_val_unseen_v1",
        "full_manifest_sha256": FULL_SHA,
        "screen_manifest_sha256": SCREEN_SHA,
        "paired_json_sha256": pair_hashes,
        "seeds": list(SEEDS),
        "full": scope(rows, ids, scenes),
        "outside_reused_screen": scope(rows, outside_ids, outside_scenes),
        "interpretation": (
            "Three paired seeds, one decode per checkpoint and episode. "
            "Training uses privileged simulator distance; this is not a "
            "deployable learned reward. Scene-and-seed intervals remain "
            "exploratory and val-unseen is development data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name]["metrics"] for name in
                      ("full", "outside_reused_screen")}, indent=2))


if __name__ == "__main__":
    main()
