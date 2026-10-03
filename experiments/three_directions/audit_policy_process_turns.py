"""Check scene isolation, replay coverage and real geodesic regression labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from collect_policy_preference_frames import digest
from collect_policy_process_turns import record_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    if manifest["schema"] != "policy_process_train_manifest_v1":
        raise ValueError("wrong manifest schema")
    result = {"schema": "policy_process_turn_audit_v1",
              "manifest_sha256": manifest_sha, "parts": {}}
    all_scenes, all_ids = set(), set()
    for part, plans in manifest["selected"].items():
        folder = args.output_root / part
        summary = json.loads((folder / "summary.json").read_text())
        if summary["manifest_sha256"] != manifest_sha or \
                summary["requested"] != len(plans) or \
                summary["completed"] != len(plans) or \
                summary["smoke_limit"] or summary["errors"]:
            raise ValueError(f"incomplete {part} collection")
        scenes, ids, episodes, regress_episodes = set(), set(), set(), set()
        counts = Counter()
        for plan in plans:
            rid = record_id(plan)
            record = json.loads((folder / "records" / f"{rid}.json").read_text())
            identity = (record["seed"], record["episode_id"], record["variant"])
            if rid != record["record_id"] or identity != (plan["seed"],
                    str(plan["episode_id"]), plan["variant"]) or \
                    record["manifest_sha256"] != manifest_sha or \
                    record["scene_id"] != plan["scene_id"] or \
                    record["terminal_mode"] != plan["terminal_mode"]:
                raise ValueError(f"record/manifest mismatch: {part}/{rid}")
            if identity in all_ids:
                raise ValueError(f"reused rollout identity: {identity}")
            ids.add(identity)
            scenes.add(record["scene_id"])
            episodes.add(record["episode_id"])
            if not (folder / record["initial_image"]).is_file():
                raise ValueError(f"missing initial image: {part}/{rid}")
            distances = [record["start_distance_to_goal_for_label_only"]]
            for turn in record["turns"]:
                if not (folder / turn["image"]).is_file() or \
                        "stop" in turn["assistant_response"].lower() or \
                        not turn["motion_actions"]:
                    raise ValueError(f"invalid image/action history: {part}/{rid}")
                distances.append(turn["distance_to_goal_for_label_only"])
            if not all(math.isfinite(x) and x >= 0 for x in distances):
                raise ValueError(f"invalid distance label: {part}/{rid}")
            if abs(distances[-1] - record["replayed_terminal_distance_m_for_audit_only"]) > 1e-5 or \
                    abs(distances[-1] - plan["terminal_distance_m_for_replay_audit_only"]) > .25:
                raise ValueError(f"terminal replay drift: {part}/{rid}")
            deltas = [after - before for before, after in zip(distances[:-1], distances[1:])]
            regressions = sum(delta >= 1.0 for delta in deltas)
            progresses = sum(delta <= -1.0 for delta in deltas)
            if regressions != record["regression_turns_geodesic_1m"] or \
                    progresses != record["progress_turns_geodesic_1m"]:
                raise ValueError(f"progress label mismatch: {part}/{rid}")
            if regressions:
                regress_episodes.add(record["episode_id"])
            counts["regression_turns_geodesic_1m"] += regressions
            counts["progress_turns_geodesic_1m"] += progresses
            counts["regression_trajectories"] += regressions > 0
            counts["motion_turns"] += len(deltas)
            counts[f"terminal:{record['terminal_mode']}"] += 1
        if all_scenes & scenes or all_ids & ids:
            raise ValueError(f"scene or identity leakage in {part}")
        all_scenes.update(scenes)
        all_ids.update(ids)
        if len(ids) != len(plans) or len(scenes) != manifest["inventory"][part]["scenes"]:
            raise ValueError(f"selection coverage mismatch in {part}")
        if summary["regression_turns_geodesic_1m"] != counts["regression_turns_geodesic_1m"] or \
                summary["progress_turns_geodesic_1m"] != counts["progress_turns_geodesic_1m"]:
            raise ValueError(f"summary label count mismatch in {part}")
        result["parts"][part] = {"trajectories": len(ids),
                                  "unique_episodes": len(episodes),
                                  "scenes": len(scenes),
                                  "episodes_with_regression": len(regress_episodes),
                                  "counts": dict(counts)}
    audit = result["parts"]["audit"]
    result["regression_sample_gate"] = {
        "require_turns_at_least": 100,
        "require_unique_episodes_at_least": 30,
        "passed": audit["counts"]["regression_turns_geodesic_1m"] >= 100 and
                  audit["episodes_with_regression"] >= 30,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["regression_sample_gate"]["passed"]:
        raise ValueError("held-out regression sample gate underpowered")


if __name__ == "__main__":
    main()
