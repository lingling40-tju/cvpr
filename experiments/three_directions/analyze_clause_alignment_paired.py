"""Paired scene-cluster comparison of frozen zero-shot clause rules."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    if probe["schema"] != "clause_alignment_zero_shot_probe_v1":
        raise ValueError("wrong source probe")
    result = {"schema": "clause_alignment_paired_analysis_v1",
              "interpretation": "exploratory train-scene paired diagnostics; no policy result",
              "probe_sha256": digest(args.probe), "partitions": {}}
    for part in ("fit", "calibration"):
        source = probe["partitions"][part]
        rows = source["per_pair"]
        if len(rows) != source["pairs"]:
            raise ValueError("pair coverage changed")
        comparisons = {}
        for method in ("final_clause", "last_two_clauses",
                       "ordered_path", "ordered_path_gain"):
            scene_diffs = defaultdict(list)
            candidate_only = control_only = both = neither = 0
            for row in rows:
                for control_margin, candidate_margin in zip(
                        row["rule_margins"]["whole"],
                        row["rule_margins"][method]):
                    a, b = int(control_margin > 0), int(candidate_margin > 0)
                    scene_diffs[row["scene_id"]].append(b - a)
                    if a and b:
                        both += 1
                    elif b:
                        candidate_only += 1
                    elif a:
                        control_only += 1
                    else:
                        neither += 1
            scenes = sorted(scene_diffs)
            rng = random.Random(11)
            boots = []
            for _ in range(5000):
                sampled = [scene_diffs[rng.choice(scenes)] for _ in scenes]
                boots.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
            boots.sort()
            count = both + candidate_only + control_only + neither
            if count != 2 * len(rows):
                raise ValueError("paired comparison coverage changed")
            comparisons[method] = {
                "comparisons": count,
                "both_correct": both,
                "method_only_correct": candidate_only,
                "whole_only_correct": control_only,
                "both_wrong": neither,
                "accuracy_change": (candidate_only - control_only) / count,
                "scene_cluster_bootstrap_change_95": [boots[125], boots[4874]],
            }
        result["partitions"][part] = comparisons
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["partitions"], indent=2))


if __name__ == "__main__":
    main()
