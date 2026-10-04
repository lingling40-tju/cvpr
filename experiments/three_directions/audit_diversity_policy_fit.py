"""Independently audit the diversity-first extra R2R-train fit replay.

Only fit records are opened. The development/audit manifest selections are
checked by identity, but their labels, images, and predictions stay closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


EXPECTED_OLD_SHA = "aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681"
EXPECTED_NEW_SHA = "bd27cac517c7909f43bca210ad8eee62456f876c56af32968e285261c58fbc37"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(plan: dict) -> tuple[int, str, int]:
    return (int(plan["seed"]), str(plan["episode_id"]), int(plan["variant"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--old-records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.manifest) != EXPECTED_NEW_SHA or \
            digest(args.old_manifest) != EXPECTED_OLD_SHA:
        raise ValueError("frozen manifest hash mismatch")
    new = json.loads(args.manifest.read_text())
    old = json.loads(args.old_manifest.read_text())
    if new["schema"] != old["schema"] != "policy_process_train_manifest_v1" or \
            new["targets"] != {"fit": 1024, "development": 320, "audit": 320} or \
            new["selected"]["fit"][:768] != old["selected"]["fit"] or \
            any(new["selected"][part] != old["selected"][part]
                for part in ("development", "audit")) or \
            {seed: source["sha256"] for seed, source in new["sources"].items()} != \
            {seed: source["sha256"] for seed, source in old["sources"].items()}:
        raise ValueError("expanded manifest changed old rows or source")
    plans = new["selected"]["fit"]
    if len({identity(plan) for plan in plans}) != 1024:
        raise ValueError("duplicate fit trajectory")
    extra = plans[768:]
    old_episodes = {str(plan["episode_id"]) for plan in plans[:768]}
    if len({str(plan["episode_id"]) for plan in extra}) != 256 or \
            any(plan["terminal_mode"] == "successfully reached the goal."
                for plan in extra):
        raise ValueError("extra selection is not diverse failed policy routes")
    folder = args.records_root / "fit"
    old_folder = args.old_records_root / "fit"
    summary = json.loads((folder / "summary.json").read_text())
    if summary["manifest_sha256"] != EXPECTED_NEW_SHA or \
            summary["requested"] != 1024 or summary["completed"] != 1024 or \
            summary["reused_from_verified_replay"] != 768 or \
            summary["smoke_limit"] or summary["errors"] or \
            summary["shards"] != 1:
        raise ValueError("fit collection incomplete or reuse count changed")
    counts, extra_counts = Counter(), Counter()
    episodes_with_regression, extra_episodes_with_regression = set(), set()
    for index, plan in enumerate(plans):
        seed, eid, variant = identity(plan)
        rid = f"s{seed}_e{eid}_v{variant}"
        record = json.loads((folder / "records" / f"{rid}.json").read_text())
        if record["schema"] != "policy_process_turn_record_v1" or \
                record["manifest_sha256"] != EXPECTED_NEW_SHA or \
                (record["seed"], str(record["episode_id"]), record["variant"]) != \
                (seed, eid, variant) or record["record_id"] != rid or \
                record["scene_id"] != plan["scene_id"] or \
                record["terminal_mode"] != plan["terminal_mode"] or \
                abs(record["source_terminal_distance_m_for_audit_only"] -
                    plan["terminal_distance_m_for_replay_audit_only"]) > 1e-5:
            raise ValueError(f"record/plan mismatch {rid}")
        if index < 768:
            old_record = json.loads((old_folder / "records" / f"{rid}.json").read_text())
            if record != dict(old_record, manifest_sha256=EXPECTED_NEW_SHA):
                raise ValueError(f"reused record drift {rid}")
        paths = [record["initial_image"]] + [turn["image"]
                                               for turn in record["turns"]]
        if not all((folder / path).is_file() for path in paths):
            raise ValueError(f"missing visual history {rid}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        if not all(math.isfinite(value) and value >= 0 for value in distances) or \
                abs(distances[-1] - record["replayed_terminal_distance_m_for_audit_only"]) > 1e-5 or \
                abs(distances[-1] - plan["terminal_distance_m_for_replay_audit_only"]) > .25:
            raise ValueError(f"invalid geodesic replay {rid}")
        deltas = [a - b for a, b in zip(distances[:-1], distances[1:])]
        forward = sum(value >= 1.0 for value in deltas)
        backward = sum(value <= -1.0 for value in deltas)
        stationary = sum(abs(value) < .1 for value in deltas)
        if forward != record["progress_turns_geodesic_1m"] or \
                backward != record["regression_turns_geodesic_1m"]:
            raise ValueError(f"stored turn labels disagree {rid}")
        target = counts if index < 768 else extra_counts
        target["trajectories"] += 1
        target["motion_turns"] += len(deltas)
        target["forward_turns_1m"] += forward
        target["regression_turns_1m"] += backward
        target["stationary_turns_0_1m"] += stationary
        target["both_direction_trajectories"] += bool(forward and backward)
        if backward:
            episodes_with_regression.add(eid)
            if index >= 768:
                extra_episodes_with_regression.add(eid)
    combined = counts + extra_counts
    if combined["regression_turns_1m"] != summary["regression_turns_geodesic_1m"] or \
            combined["forward_turns_1m"] != summary["progress_turns_geodesic_1m"]:
        raise ValueError("fit label counts disagree with collection summary")
    extra_gate = {"minimum_regression_turns": 100,
                  "minimum_episode_ids_with_regression": 30,
                  "passed": extra_counts["regression_turns_1m"] >= 100 and
                            len(extra_episodes_with_regression) >= 30}
    report = {
        "schema": "diversity_policy_fit_replay_audit_v1",
        "interpretation": "R2R-train fit-only replay/label coverage; no representation or navigation result.",
        "new_manifest_sha256": EXPECTED_NEW_SHA,
        "old_manifest_sha256": EXPECTED_OLD_SHA,
        "old_fit": {"episodes": len(old_episodes), "counts": dict(counts)},
        "extra_fit": {"trajectories": 256,
                      "episode_ids": len({str(plan["episode_id"]) for plan in extra}),
                      "new_episode_ids": len({str(plan["episode_id"]) for plan in extra} - old_episodes),
                      "episode_ids_with_regression": len(extra_episodes_with_regression),
                      "terminal_modes": dict(Counter(plan["terminal_mode"] for plan in extra)),
                      "counts": dict(extra_counts)},
        "combined_fit": {"trajectories": 1024,
                         "episode_ids": len({str(plan["episode_id"]) for plan in plans}),
                         "episode_ids_with_regression": len(episodes_with_regression),
                         "counts": dict(combined)},
        "extra_sample_gate": extra_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
