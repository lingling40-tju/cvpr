"""Recount a frozen positive-trajectory screen without importing its analyzer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


MANIFEST_HASHES = {
    "development": "8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3",
    "reserved": "412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"metric differs: {actual} vs {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validators", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.report.read_text())
    role = manifest["role"]
    if role not in MANIFEST_HASHES or sha(args.manifest) != MANIFEST_HASHES[role]:
        raise ValueError("frozen manifest differs")
    ids = [str(value) for value in manifest["episode_ids"]]
    expected = dict(zip(ids, manifest["scene_ids"]))
    rows = [json.loads(line) for line in args.compact.read_text().splitlines()]
    if len(ids) != 256 or len(expected) != 256 or len(rows) != 256 or \
            len({str(row["episode_id"]) for row in rows}) != 256:
        raise ValueError("unique episode coverage differs")
    if {str(row["episode_id"]): row["scene_id"] for row in rows} != expected:
        raise ValueError("episode or scene identity differs")
    if len(set(expected.values())) != 8 or report["schema"] != "positive_trajectory_train_scene_pair_v1" or \
            report["role"] != role or report["split"] != "train" or \
            report["episodes"] != 256 or report["scenes"] != 8 or \
            report["manifest_sha256"] != MANIFEST_HASHES[role] or \
            report["compact_sha256"] != sha(args.compact) or report["inference_errors"] != 0:
        raise ValueError("report provenance differs")
    totals = {arm: [0, 0.0] for arm in ("control", "candidate")}
    clusters = defaultdict(lambda: [0, 0, 0.0])
    discordant = [0, 0]
    for row in rows:
        for arm in totals:
            success, spl = row[f"{arm}_success"], row[f"{arm}_spl"]
            if success not in (0, 1) or isinstance(spl, bool) or \
                    not math.isfinite(spl) or not 0 <= spl <= 1:
                raise ValueError("invalid episode metric")
            totals[arm][0] += success
            totals[arm][1] += spl
        ds = row["candidate_success"] - row["control_success"]
        dp = row["candidate_spl"] - row["control_spl"]
        bucket = clusters[row["scene_id"]]
        bucket[0] += 1
        bucket[1] += ds
        bucket[2] += dp
        discordant[0] += ds == 1
        discordant[1] += ds == -1
    for arm, (successes, spl_sum) in totals.items():
        validator = json.loads((args.validators / f"{report[arm]}.validated.json").read_text())
        if validator["label"] != report[arm] or validator["episodes"] != 256 or \
                validator["inference_errors"] != 0 or validator["successes"] != successes:
            raise ValueError("arm validator differs")
        equal(validator["spl"], spl_sum / 256)
        equal(report[f"{arm}_successes"], successes)
        equal(report[f"{arm}_sr"], successes / 256)
        equal(report[f"{arm}_spl"], spl_sum / 256)
    sr = 100 * (totals["candidate"][0] - totals["control"][0]) / 256
    spl = 100 * (totals["candidate"][1] - totals["control"][1]) / 256
    equal(report["paired_sr_points"], sr)
    equal(report["paired_spl_points"], spl)
    if discordant != [report["candidate_only_success"], report["control_only_success"]]:
        raise ValueError("paired discordance differs")
    rng = random.Random(20261006128)
    names = sorted(clusters)
    draws = [[], []]
    for _ in range(10000):
        selected = [clusters[name] for name in rng.choices(names, k=8)]
        n = sum(value[0] for value in selected)
        for index in (0, 1):
            draws[index].append(100 * sum(value[index + 1] for value in selected) / n)
    for index, metric in enumerate(("sr", "spl")):
        values = sorted(draws[index])
        for actual, expected_value in zip((values[249], values[9749]),
                                         report[f"scene_bootstrap_95pct_{metric}_points"]):
            equal(actual, expected_value)
    if args.gate:
        gate = json.loads(args.gate.read_text())
        if role != "development" or gate["schema"] != "positive_trajectory_frozen_development_gate_v1" or \
                gate["requires_each_metric_points_at_least"] != 2.0 or \
                gate["pass"] != (sr >= 2.0 and spl >= 2.0) or gate["reserved_screen_opened"] is not False:
            raise ValueError("frozen development gate differs")
        equal(gate["paired_sr_points"], sr)
        equal(gate["paired_spl_points"], spl)
    result = {"schema": "positive_trajectory_compact_recount_v1", "role": role,
              "episodes": 256, "scenes": 8, "inference_errors": 0,
              "control_successes": totals["control"][0],
              "candidate_successes": totals["candidate"][0],
              "paired_sr_points": sr, "paired_spl_points": spl,
              "compact_sha256": sha(args.compact), "bootstrap_recount_agrees": True}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
