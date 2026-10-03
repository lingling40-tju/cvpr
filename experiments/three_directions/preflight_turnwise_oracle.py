"""CPU-only upper-bound check for action-level progress credit on n=4 groups.

Reads previously replayed R2R-train trajectories. The simulator distances
are diagnostic labels and are never a proposed inference-time policy input.
The locked audit split is deliberately not read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


MANIFEST_SHA = "a99a15020b3d7ffe061ea82ab830e4d617e5ad8f90e3fff8979ab80079a26356"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_split(rows: list[dict], folder: Path) -> dict:
    groups = defaultdict(list)
    counts = Counter()
    scene_ids = set()
    max_abs_normalized_turn_reward = 0.0
    for source in rows:
        seed, eid, variant = source["seed"], str(source["episode_id"]), source["variant"]
        name = f"s{seed}_e{eid}_v{variant}"
        record = json.loads((folder / "records" / f"{name}.json").read_text())
        if record["record_id"] != name or record["manifest_sha256"] != MANIFEST_SHA:
            raise ValueError(f"record identity mismatch: {name}")
        if record["scene_id"] != source["scene_id"] or \
                record["terminal_mode"] != source["terminal_mode"]:
            raise ValueError(f"scene or terminal mode mismatch: {name}")
        start = float(record["start_distance_to_goal_for_label_only"])
        distances = [start] + [float(t["distance_to_goal_for_label_only"])
                                for t in record["turns"]]
        if not distances or not all(math.isfinite(d) and d >= 0 for d in distances):
            raise ValueError(f"invalid distance trace: {name}")
        if abs(distances[-1] - float(record["source_terminal_distance_m_for_audit_only"])) > 1e-4:
            raise ValueError(f"terminal replay drift: {name}")
        if any(not turn["motion_actions"] for turn in record["turns"]):
            raise ValueError(f"empty motion turn: {name}")
        deltas = [before - after for before, after in zip(distances, distances[1:])]
        normalizer = max(start, 3.0)
        reward = [delta / normalizer for delta in deltas]
        if abs(sum(deltas) - (start - distances[-1])) > 1e-6:
            raise ValueError(f"progress does not telescope: {name}")
        if abs(sum(reward) - (start - distances[-1]) / normalizer) > 1e-6:
            raise ValueError(f"normalized reward does not telescope: {name}")
        max_abs_normalized_turn_reward = max(
            max_abs_normalized_turn_reward, *(abs(value) for value in reward))
        counts["motion_turns"] += len(reward)
        counts["forward_turns_geodesic_0p1m"] += sum(d > 0.1 for d in deltas)
        counts["regression_turns_geodesic_0p1m"] += sum(d < -0.1 for d in deltas)
        counts["turns_over_1m_geodesic_change"] += sum(abs(d) > 1.0 for d in deltas)
        groups[(seed, eid)].append({"variant": variant, "scene": source["scene_id"],
                                    "mode": source["terminal_mode"],
                                    "reward": reward, "terminal_progress": (start - distances[-1]) / normalizer})
        scene_ids.add(source["scene_id"])
        counts["trajectories"] += 1
    for key, items in groups.items():
        if len(items) != 4 or {x["variant"] for x in items} != {0, 1, 2, 3} or \
                len({x["scene"] for x in items}) != 1:
            raise ValueError(f"incomplete four-rollout group: {key}")
        counts["groups"] += 1
        if any(x["mode"] == "successfully reached the goal." for x in items):
            counts["groups_with_success"] += 1
            continue
        counts["all_failure_groups"] += 1
        scalar = [x["terminal_progress"] for x in items]
        scalar_mean = sum(scalar) / 4
        scalar_sign = [(x > scalar_mean) - (x < scalar_mean) for x in scalar]
        returns = []
        for item in items:
            running = 0.0
            values = []
            for reward in reversed(item["reward"]):
                running += reward
                values.append(running)
            returns.append(list(reversed(values)))
        max_turn = max(map(len, returns))
        group_shift = False
        group_nonzero = False
        for turn in range(max_turn):
            active = [i for i, values in enumerate(returns) if turn < len(values)]
            if len(active) < 2:
                continue
            mean = sum(returns[i][turn] for i in active) / len(active)
            for i in active:
                value = returns[i][turn] - mean
                sign = (value > 1e-8) - (value < -1e-8)
                counts["comparable_action_turns"] += 1
                counts["nonzero_turnwise_advantages"] += sign != 0
                counts["turn_sign_differs_from_terminal"] += sign != 0 and sign != scalar_sign[i]
                group_shift |= sign != 0 and sign != scalar_sign[i]
                group_nonzero |= sign != 0
        counts["groups_with_turnwise_signal"] += group_nonzero
        counts["groups_with_credit_shift"] += group_shift
    return {"scenes": len(scene_ids),
            "max_abs_normalized_turn_reward": max_abs_normalized_turn_reward,
            **dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.manifest) != MANIFEST_SHA:
        raise ValueError("group-four manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    report = {
        "schema": "turnwise_oracle_train_preflight_v1",
        "manifest_sha256": MANIFEST_SHA,
        "reward_formula": "(d_before-d_after)/max(d_start,3m); telescopes exactly",
        "discount": 1.0,
        "fit": analyze_split(manifest["selected"]["fit"], args.records_root / "fit"),
        "development": analyze_split(manifest["selected"]["development"], args.records_root / "development"),
        "interpretation": (
            "Replayed train-scene oracle-label upper bound and credit-assignment diagnostic only; "
            "the locked audit was not read, no policy was trained, and no navigation gain is implied."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
