"""Aggregate three fixed reserved-screen pairs after independent recounts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


MANIFEST_SHA = "412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241"
SEEDS = (11, 22, 33)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError("reserved manifest differs")
    manifest = json.loads(args.manifest.read_text())
    mapping = dict(zip(map(str, manifest["episode_ids"]), manifest["scene_ids"]))
    if manifest["role"] != "reserved" or len(mapping) != 256 or len(set(mapping.values())) != 8:
        raise ValueError("reserved coverage differs")
    pairs = []
    clusters = {}
    for seed in SEEDS:
        path = args.root / f"seed{seed}_reserved_paired.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        report = json.loads((args.root / f"seed{seed}_reserved_pair.json").read_text())
        independent = json.loads((args.root / f"seed{seed}_independent_recount.json").read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(rows) != 256 or {str(row["episode_id"]): row["scene_id"] for row in rows} != mapping:
            raise ValueError("seed episode or scene mapping differs")
        if report["compact_sha256"] != digest or independent["compact_sha256"] != digest or \
                report["role"] != "reserved" or report["manifest_sha256"] != MANIFEST_SHA or \
                independent["role"] != "reserved" or independent["episodes"] != 256 or \
                independent["scenes"] != 8 or independent["inference_errors"] != 0 or \
                independent["bootstrap_recount_agrees"] is not True or \
                report["control"] != f"positive_trajectory_control_128step_seed{seed}" or \
                report["candidate"] != f"positive_trajectory_candidate_128step_seed{seed}":
            raise ValueError("seed provenance differs")
        grouped = defaultdict(lambda: [0, 0, 0.0])
        for row in rows:
            for arm in ("control", "candidate"):
                if row[f"{arm}_success"] not in (0, 1) or \
                        not math.isfinite(row[f"{arm}_spl"]) or not 0 <= row[f"{arm}_spl"] <= 1:
                    raise ValueError("invalid metric")
            bucket = grouped[row["scene_id"]]
            bucket[0] += 1
            bucket[1] += row["candidate_success"] - row["control_success"]
            bucket[2] += row["candidate_spl"] - row["control_spl"]
        clusters[seed] = grouped
        sr = 100 * statistics.mean(row["candidate_success"] - row["control_success"] for row in rows)
        spl = 100 * statistics.mean(row["candidate_spl"] - row["control_spl"] for row in rows)
        for metric, value in (("sr", sr), ("spl", spl)):
            for reference in (report, independent):
                if not math.isclose(reference[f"paired_{metric}_points"], value, abs_tol=1e-10):
                    raise ValueError("seed recount differs")
        pairs.append({"seed": seed, "control_successes": sum(row["control_success"] for row in rows),
                      "candidate_successes": sum(row["candidate_success"] for row in rows),
                      "paired_sr_points": sr, "paired_spl_points": spl, "compact_sha256": digest})
    rng = random.Random(20261007128)
    names = sorted(set(mapping.values()))
    draws = [[], []]
    for _ in range(10000):
        sampled_seeds = rng.choices(SEEDS, k=3)
        sampled_scenes = rng.choices(names, k=8)
        for index in (0, 1):
            seed_means = []
            for seed in sampled_seeds:
                buckets = [clusters[seed][scene] for scene in sampled_scenes]
                seed_means.append(sum(bucket[index + 1] for bucket in buckets) /
                                  sum(bucket[0] for bucket in buckets))
            draws[index].append(100 * statistics.mean(seed_means))
    result = {"schema": "positive_trajectory_three_seed_reserved_v1", "seeds": list(SEEDS),
              "episodes_per_seed": 256, "scenes": 8, "inference_errors": 0,
              "manifest_sha256": MANIFEST_SHA, "training_steps_each_arm": 128,
              "group_size": 4, "pairs": pairs,
              "interpretation": "Prospective RL-stage reserved train-scene screen after a one-seed development gate; prior SFT/exploration may have used these train scenes. Three-seed intervals are descriptive, not proof of robust generalization."}
    for index, metric in enumerate(("sr", "spl")):
        values = [pair[f"paired_{metric}_points"] for pair in pairs]
        ordered = sorted(draws[index])
        result[f"mean_paired_{metric}_points"] = statistics.mean(values)
        result[f"sample_sd_paired_{metric}_points"] = statistics.stdev(values)
        result[f"seed_and_scene_bootstrap_95pct_{metric}_points"] = [ordered[249], ordered[9749]]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
