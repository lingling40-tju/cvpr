"""Train-rollout diagnostic for the frozen stop-aware reward.

The 3.5 m boundary is descriptive and is not used to tune this reward.
This script never reads val-unseen results or images.
"""

import argparse
import hashlib
import json
from pathlib import Path


def auc(positive, negative):
    if not positive or not negative:
        return None
    wins = sum(a > b for a in positive for b in negative)
    ties = sum(a == b for a in positive for b in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hasher = hashlib.sha256()
    stops = []
    total = 0
    for line in args.rollout.open("rb"):
        hasher.update(line)
        record = json.loads(line)
        for row in record["info"]:
            total += 1
            if row["end_reason"] == "stopped but goal not reached.":
                stops.append(row)
    assert total == 1024 and len(stops) == 328
    near = [row for row in stops if row["distance_to_goal"] <= 3.5]
    far = [row for row in stops if row["distance_to_goal"] > 3.5]
    rewarded = [row for row in stops if row["fused_reward"]["applied_bonus"] > 0]
    assert len(near) + len(far) == len(stops)
    assert len(rewarded) == 183
    raw_auc = auc([row["fused_reward"]["raw"] for row in near],
                  [row["fused_reward"]["raw"] for row in far])
    report = {
        "schema": "stopaware_train_stop_calibration_v1",
        "interpretation": "On-policy train-scene diagnostic only; the 3.5 m boundary is descriptive, and no held-out navigation effect is inferred.",
        "rollout_sha256": hasher.hexdigest(),
        "rollouts": total,
        "unsuccessful_voluntary_stops": len(stops),
        "stops_within_3_5_m": len(near),
        "stops_beyond_3_5_m": len(far),
        "stops_with_positive_bonus": len(rewarded),
        "positive_bonus_stops_within_3_5_m": sum(row["distance_to_goal"] <= 3.5 for row in rewarded),
        "raw_score_near_vs_far_auc": raw_auc,
        "raw_score_near_mean": sum(row["fused_reward"]["raw"] for row in near) / len(near),
        "raw_score_far_mean": sum(row["fused_reward"]["raw"] for row in far) / len(far),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
