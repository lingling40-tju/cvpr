"""Independently audit frozen crossed-goal fit labels against RGB replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from collect_cross_goal_fit_labels import CHANGE_M, cross_count
from prepare_cross_goal_fit_manifest import IDS_SHA, RENDER_SHA, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.ids) != IDS_SHA:
        raise ValueError("frozen ID source changed")
    ids = json.loads(args.ids.read_text())
    manifest = json.loads(args.manifest.read_text())
    summary_path = args.label_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    source_ids = {str(row["episode_id"]): row["scene_id"]
                  for row in ids["rows"]}
    manifest_sha = digest(args.manifest)
    if (manifest["schema"] != "cross_goal_fit_manifest_v1"
            or manifest["source_sha256"]["ids"] != IDS_SHA
            or manifest["source_sha256"]["render_manifest"] != RENDER_SHA
            or summary["schema"] != "cross_goal_fit_label_summary_v1"
            or summary["manifest_sha256"] != manifest_sha
            or summary["requested"] != summary["completed"]
            != manifest["trajectories"]):
        raise ValueError("manifest/label summary mismatch")
    seen = set()
    crossed = 0
    crossed_ids = set()
    largest_drift = 0.0
    for plan in manifest["plans"]:
        eid, variant = str(plan["episode_id"]), int(plan["variant"])
        oid = str(plan["wrong_episode_id"])
        rid = f"s{plan['seed']}_e{eid}_v{variant}"
        if ((eid, variant) in seen or eid == oid
                or source_ids.get(eid) != plan["scene_id"]
                or source_ids.get(oid) != plan["scene_id"]):
            raise ValueError(f"fit ID or wrong goal contamination: {rid}")
        seen.add((eid, variant))
        record = json.loads((args.label_root / "records" / f"{rid}.json").read_text())
        rgb = json.loads((args.render_root / "fit" / "records" /
                          f"{rid}.json").read_text())
        if (record["schema"] != "cross_goal_fit_label_record_v1"
                or record["manifest_sha256"] != manifest_sha
                or record["render_manifest_sha256"] != RENDER_SHA
                or rgb["manifest_sha256"] != RENDER_SHA
                or record["record_id"] != rgb["record_id"] != rid
                or record["wrong_episode_id"] != oid):
            raise ValueError(f"record source mismatch: {rid}")
        correct = record["correct_distance_m_for_label_only"]
        wrong = record["wrong_distance_m_for_label_only"]
        expected = [rgb["start_distance_to_goal_for_label_only"]] + [
            row["distance_to_goal_for_label_only"] for row in rgb["turns"]]
        if (len(correct) != len(wrong) or len(correct) != len(expected)
                or not all(math.isfinite(float(v)) and v >= 0
                           for v in correct + wrong)):
            raise ValueError(f"nonfinite or incomplete goal trace: {rid}")
        drift = max(abs(a - b) for a, b in zip(correct, expected))
        if drift > 1e-4 or record["max_correct_distance_drift_m"] > 1e-4:
            raise ValueError(f"correct-goal replay drift: {rid}")
        value = cross_count(correct, wrong)
        if value != record["crossed_turns_0p5m"]:
            raise ValueError(f"crossed-turn count mismatch: {rid}")
        crossed += value
        largest_drift = max(largest_drift, drift)
        if value:
            crossed_ids.add(eid)
    if (len(seen) != manifest["trajectories"]
            or summary["crossed_turns_0p5m"] != crossed
            or summary["episode_ids_with_crossed_turns"] != len(crossed_ids)
            or summary["unique_episode_ids"] != manifest["episode_ids"]
            or summary["max_correct_distance_drift_m"] > 1e-4):
        raise ValueError("crossed-goal aggregate mismatch")
    report = {
        "schema": "cross_goal_fit_label_audit_v1",
        "interpretation": "Train-fit geodesic supervision only; no learned reward or navigation result",
        "source_sha256": {"ids": IDS_SHA, "manifest": manifest_sha,
                          "summary": digest(summary_path)},
        "trajectories": len(seen), "episode_ids": manifest["episode_ids"],
        "crossed_turns_0p5m": crossed,
        "episode_ids_with_crossed_turns": len(crossed_ids),
        "maximum_correct_distance_drift_m": largest_drift,
        "change_threshold_m": CHANGE_M,
        "sample_gate": summary["sample_gate"],
        "label_replay_seconds": summary["elapsed_seconds"],
        "reachability_preflight_seconds": manifest["elapsed_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
