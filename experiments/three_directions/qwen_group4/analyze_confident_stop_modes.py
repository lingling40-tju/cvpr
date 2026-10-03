"""Post hoc paired termination-mode audit for the confidence pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unforced_failure(row: dict, arm: str) -> bool:
    item = row[arm]
    return not item["success"] and item["early_stop_reason"] is None


def scene_interval(rows: list[dict], draws: int = 10000) -> list[float]:
    scenes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        scenes[row["scene_id"]].append(row)
    names = sorted(scenes)
    rng = random.Random(20261004)
    values = []
    for _ in range(draws):
        sampled = [row for name in rng.choices(names, k=len(names))
                   for row in scenes[name]]
        values.append(100 * sum(
            int(unforced_failure(row, "candidate")) -
            int(unforced_failure(row, "control"))
            for row in sampled) / len(sampled))
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.paired.read_text().splitlines()]
    manifest = json.loads(args.manifest.read_text())
    analysis = json.loads(args.analysis.read_text())
    ids = list(map(str, manifest["episode_ids"]))
    scenes = list(map(str, manifest["scene_ids"]))
    if len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError("invalid episode manifest")
    if len(rows) != 256 or [row["episode_id"] for row in rows] != ids or \
            [row["scene_id"] for row in rows] != scenes:
        raise ValueError("paired rows do not match manifest")
    if digest(args.manifest) != analysis["manifest_sha256"] or \
            analysis["candidate"] != "qwen_confident_64_seed11" or \
            analysis["control"] != "qwen_exact_control_64_seed11":
        raise ValueError("analysis identity mismatch")
    if any(arm["early_stop_reason"] == "inference_error"
           for row in rows for arm in (row["candidate"], row["control"])):
        raise ValueError("inference-error row in compact pair")
    counts = {}
    for arm in ("candidate", "control"):
        values = [row[arm] for row in rows]
        expected = analysis[f"{arm}_metrics"]
        if expected["count"] != 256 or expected["inference_errors"] != 0 or \
                sum(bool(item["success"]) for item in values) != expected["successes"] or \
                not math.isclose(sum(float(item["spl"]) for item in values) / 256,
                                 expected["spl"], abs_tol=1e-12):
            raise ValueError(f"{arm} metrics disagree with summary")
        by_outcome = Counter(
            f"{'success' if item['success'] else 'failure'}:{item['early_stop_reason']}"
            for item in values)
        counts[arm] = {
            "successes": expected["successes"],
            "unforced_failures": sum(unforced_failure(row, arm) for row in rows),
            "outcome_by_evaluator_stop_reason": dict(sorted(by_outcome.items())),
        }
    candidate_only = sum(unforced_failure(row, "candidate") and
                         not unforced_failure(row, "control") for row in rows)
    control_only = sum(unforced_failure(row, "control") and
                       not unforced_failure(row, "candidate") for row in rows)
    difference = 100 * (counts["candidate"]["unforced_failures"] -
                        counts["control"]["unforced_failures"]) / 256
    report = {
        "schema": "confidence_val256_termination_diagnostic_v1",
        "source_sha256": {
            "paired_episodes": digest(args.paired),
            "manifest": digest(args.manifest),
            "paired_analysis": digest(args.analysis),
        },
        "episodes": 256,
        "scenes": len(set(scenes)),
        "arms": counts,
        "paired_unforced_failure": {
            "candidate_only": candidate_only,
            "control_only": control_only,
            "difference_pp": difference,
            "scene_bootstrap95_pp": scene_interval(rows),
            "draws": 10000,
            "seed": 20261004,
        },
        "success_discordance_with_other_arm_unforced_failure": {
            "control_success_candidate_unforced_failure": sum(
                row["control"]["success"] and unforced_failure(row, "candidate")
                for row in rows),
            "candidate_success_control_unforced_failure": sum(
                row["candidate"]["success"] and unforced_failure(row, "control")
                for row in rows),
        },
        "interpretation": (
            "Post hoc reused val-unseen development diagnostic. A missing evaluator "
            "forced-stop reason is consistent with model-selected STOP, but is not "
            "an independent action trace or a causal attribution of the SR gap."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
