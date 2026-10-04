"""Inventory missed near-goal STOP opportunities in completed n=4 rollouts.

This uses privileged per-turn distances only as a training-data diagnostic.
It does not infer that any reward caused the behavior or predict val-unseen SR.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def inspect(path: Path, seed: int) -> dict:
    counts = Counter()
    end_reasons = Counter()
    missed_reasons = Counter()
    episode_ids = set()
    near_failure_ids = set()
    crossing_ids = set()
    with path.open() as stream:
        for step, line in enumerate(stream, 1):
            record = json.loads(line)
            if record["step"] != step or len(record["info"]) != 16:
                raise ValueError(f"seed {seed}: bad step or group at {step}")
            groups = defaultdict(list)
            for item in record["info"]:
                groups[str(item["episode_id"])].append(item)
            if len(groups) != 4 or set(map(len, groups.values())) != {4} or \
                    episode_ids.intersection(groups):
                raise ValueError(f"seed {seed}: duplicate/non-n4 groups")
            episode_ids.update(groups)
            for items in groups.values():
                counts["episode_groups"] += 1
                all_failure = not any(item["task_success"] for item in items)
                counts["all_failure_groups"] += all_failure
                missed_in_group = False
                for item in items:
                    turns = item["gen_traj"]
                    start = float(item["oracle_start_distance"])
                    if not turns or not math.isfinite(start) or start < 0:
                        raise ValueError("missing oracle trajectory")
                    distances = [start]
                    for turn in turns:
                        before = float(turn["oracle_before_distance"])
                        after = float(turn["oracle_after_distance"])
                        if not math.isfinite(before) or not math.isfinite(after) or \
                                min(before, after) < 0 or \
                                abs(before - distances[-1]) > 1e-4:
                            raise ValueError("discontinuous oracle distance")
                        distances.append(after)
                        if before > 3.0 >= after:
                            counts["entered_3m_turns"] += 1
                            crossing_ids.add(str(item["episode_id"]))
                        if before <= 3.0 < after:
                            counts["left_3m_turns"] += 1
                    final = float(item["distance_to_goal"])
                    if not math.isfinite(final) or final < 0 or \
                            abs(final - distances[-1]) > 1e-4:
                        raise ValueError("terminal distance mismatch")
                    success = bool(item["task_success"])
                    counts["rollouts"] += 1
                    counts["successes"] += success
                    end_reasons[str(item.get("end_reason"))] += 1
                    if not success:
                        counts["failures"] += 1
                        close_at_any = min(distances) <= 3.0
                        close_before_final = min(distances[:-1]) <= 3.0
                        counts["failed_ever_within_3m"] += close_at_any
                        if close_at_any:
                            near_failure_ids.add(str(item["episode_id"]))
                            missed_reasons[str(item.get("end_reason"))] += 1
                        counts["failed_within_3m_before_final"] += close_before_final
                        counts["failed_entered_3m_then_left"] += \
                            close_before_final and final > 3.0
                        counts["failed_final_within_3m"] += final <= 3.0
                        counts["failed_final_3_to_4m"] += 3.0 <= final < 4.0
                        counts["failed_turn_count_ge_10"] += len(turns) >= 10
                        missed_in_group |= close_at_any
                counts["all_failure_groups_with_near_goal_failure"] += \
                    all_failure and missed_in_group
    if step != 128 or len(episode_ids) != 512 or \
            counts["rollouts"] != 2048 or counts["episode_groups"] != 512:
        raise ValueError(f"seed {seed}: incomplete n4 training rollout")
    return {"seed": seed, "source": str(path),
            "counts": dict(counts), "end_reasons": dict(end_reasons),
            "near_goal_failure_unique_episode_ids": len(near_failure_ids),
            "three_meter_entry_unique_episode_ids": len(crossing_ids),
            "near_goal_failure_end_reasons": dict(missed_reasons)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for seed in (11, 22, 33):
        path = args.checkpoint_root / f"oracle_turnwise_exact512_128_seed{seed}" / "rollout.jsonl"
        reports.append(inspect(path, seed))
    result = {"schema": "oracle_exact512_stop_boundary_train_preflight_v1",
              "group_size": 4, "seeds": [11, 22, 33],
              "reports": reports,
              "interpretation": "Training-only privileged distance inventory; no causal or navigation claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
