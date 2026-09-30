"""Paired descriptive analysis of the fixed val-unseen subset.

Intervals quantify only variation over these selected episodes; they do not
correct the small, non-random subset or single training seed.
"""

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "val_unseen16"
ARMS = ("sft", "control", "event")


def load():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    ids = manifest["episode_ids"]
    rows = {}
    for arm in ARMS:
        summary = json.loads((ROOT / arm / "summary.json").read_text())
        assert summary["episode_ids"] == ids and summary["count"] == len(ids)
        arm_rows = {}
        for episode_id in ids:
            path = ROOT / arm / "log" / f"stats_{episode_id}_0.json"
            row = json.loads(path.read_text())
            assert str(row["id"]) == episode_id
            arm_rows[episode_id] = row
        rows[arm] = arm_rows
    return ids, rows


def bootstrap_ci(values, seed=20261001, repetitions=10000):
    rng = random.Random(seed)
    sample_means = []
    n = len(values)
    for _ in range(repetitions):
        sample_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    sample_means.sort()
    return [sample_means[int(.025 * repetitions)], sample_means[int(.975 * repetitions)]]


def main():
    ids, rows = load()
    results = {"split": "val_unseen", "episode_count": len(ids), "arms": {}, "paired": {}}
    for arm in ARMS:
        values = [rows[arm][episode_id] for episode_id in ids]
        results["arms"][arm] = {
            "successes": sum(bool(row["success"]) for row in values),
            "sr": sum(bool(row["success"]) for row in values) / len(values),
            "spl": sum(float(row["spl"]) for row in values) / len(values),
            "inference_errors": sum(row.get("early_stop_reason") == "inference_error" for row in values),
        }
    for baseline in ("sft", "control"):
        sr_diff = [float(bool(rows["event"][i]["success"])) - float(bool(rows[baseline][i]["success"]))
                   for i in ids]
        spl_diff = [float(rows["event"][i]["spl"]) - float(rows[baseline][i]["spl"])
                    for i in ids]
        results["paired"][f"event_minus_{baseline}"] = {
            "sr_difference": sum(sr_diff) / len(ids),
            "sr_95pct_bootstrap_ci": bootstrap_ci(sr_diff),
            "spl_difference": sum(spl_diff) / len(ids),
            "spl_95pct_bootstrap_ci": bootstrap_ci(spl_diff),
            "event_only_successes": sum(value == 1 for value in sr_diff),
            "baseline_only_successes": sum(value == -1 for value in sr_diff),
        }
    (ROOT / "paired_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
