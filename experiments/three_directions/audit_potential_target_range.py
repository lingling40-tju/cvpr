"""Audit whether a bounded potential can match local progress targets.

This is a fit-only label diagnostic, not a navigation or model metric.
Intermediate motion changes without a forward/backward/stationary label
break a contiguous constraint block because the model may change its
potential arbitrarily there.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MANIFEST_SHA = "43bf8bc8af46f07051d299810c1dd2975034a0b84da7f278b98fda5db761006c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.records.glob("*.json"))
    if len(files) != 512:
        raise ValueError(f"expected 512 rendered fit records, got {len(files)}")
    strict_over_two = 0
    zero_gap_over_two = 0
    max_strict_range = 0.0
    blocks = 0
    for path in files:
        record = json.loads(path.read_text())
        if (record["schema"] != "policy_process_turn_record_v1"
                or record["manifest_sha256"] != MANIFEST_SHA):
            raise ValueError(f"wrong fit record source: {path}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        block, all_values = [0.0], [0.0]
        state = total = 0.0
        largest = 0.0
        for before, after in zip(distances, distances[1:]):
            delta = before - after
            if abs(delta) >= 1.0:
                change = math.tanh(delta / 3.0)
                state += change
                total += change
                block.append(state)
            elif abs(delta) < .1:
                block.append(state)
            else:
                largest = max(largest, max(block) - min(block))
                blocks += 1
                state, block = 0.0, [0.0]
            all_values.append(total)
        largest = max(largest, max(block) - min(block))
        blocks += 1
        strict_over_two += largest > 2.0
        zero_gap_over_two += max(all_values) - min(all_values) > 2.0
        max_strict_range = max(max_strict_range, largest)
    report = {
        "schema": "cross_goal_bounded_potential_target_range_audit_v1",
        "interpretation": "Fit-only target compatibility, not model accuracy or navigation performance",
        "manifest_sha256": MANIFEST_SHA,
        "trajectories": len(files),
        "contiguous_labeled_blocks": blocks,
        "trajectories_with_contiguous_target_range_gt_2": strict_over_two,
        "trajectories_with_target_range_gt_2_if_unlabeled_motion_is_zero": zero_gap_over_two,
        "maximum_contiguous_target_range": max_strict_range,
        "bounded_potential_range": [-1.0, 1.0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
