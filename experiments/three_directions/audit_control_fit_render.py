"""Cross-check selected RGB replay against prior label-only fit replay."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_IDS_SHA = "d214dc38cf8d4094a6329afd73c080e60adaa0a6b6e30b69cb92f335aecae081"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--rgb-seconds-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.ids) != EXPECTED_IDS_SHA:
        raise ValueError("frozen ID plan changed")
    ids = json.loads(args.ids.read_text())
    source = json.loads(args.label_manifest.read_text())
    selected = json.loads(args.render_manifest.read_text())
    selection = json.loads(args.selection_report.read_text())
    summary = json.loads((args.render_root / "fit" / "summary.json").read_text())
    label_summary = json.loads((args.labels_root / "summary.json").read_text())
    selected_sha = digest(args.render_manifest)
    source_sha = digest(args.label_manifest)
    valid_count = len(source["selected"]["fit"])
    if (source["schema"] != "policy_process_train_manifest_v1" or
            selected["schema"] != "policy_process_train_manifest_v1" or
            valid_count < 1000 or
            source["targets"] != {"fit": valid_count,
                                  "development": 0, "audit": 0} or
            selected["targets"] != {"fit": 512, "development": 0, "audit": 0} or
            selected["source_all_variant_manifest_sha256"] != source_sha or
            selected["sources"] != source["sources"] or
            selected["selected"]["development"] or selected["selected"]["audit"] or
            selection["schema"] != "control_fit_render_selection_v1" or
            selection["source_sha256"]["ids"] != EXPECTED_IDS_SHA or
            selection["source_sha256"]["all_variant_manifest"] != source_sha or
            selection["source_sha256"]["label_only_summary"] !=
            digest(args.labels_root / "summary.json") or
            selection["sample_gate"]["passed"] is not True or
            selection["render_manifest_sha256"] != selected_sha or
            summary["schema"] != "policy_process_turn_collection_v1" or
            summary["manifest_sha256"] != selected_sha or
            summary["requested"] != 512 or summary["completed"] != 512 or
            summary["errors"] or summary["smoke_limit"] or
            label_summary["requested"] != valid_count or
            label_summary["completed"] != valid_count):
        raise ValueError("selected RGB replay source incomplete")
    expected_ids = {str(row["episode_id"]) for row in ids["rows"]}
    if len(expected_ids) != 256:
        raise ValueError("frozen ID count changed")
    counts = Counter()
    per_episode = Counter()
    max_turn_drift = 0.0
    record_root = args.render_root / "fit"
    for plan in selected["selected"]["fit"]:
        eid = str(plan["episode_id"])
        rid = f"s{plan['seed']}_e{eid}_v{plan['variant']}"
        if eid not in expected_ids:
            raise ValueError(f"unfrozen episode in selected replay {eid}")
        label = json.loads((args.labels_root / "records" /
                            f"{rid}.json").read_text())
        record = json.loads((record_root / "records" /
                             f"{rid}.json").read_text())
        if label["schema"] != "control_fit_label_only_replay_v1" or \
                label["manifest_sha256"] != source_sha or \
                record["schema"] != "policy_process_turn_record_v1" or \
                record["manifest_sha256"] != selected_sha or \
                label["record_id"] != rid or record["record_id"] != rid or \
                label["scene_id"] != plan["scene_id"] or \
                record["scene_id"] != plan["scene_id"] or \
                label["terminal_mode"] != plan["terminal_mode"] or \
                record["terminal_mode"] != plan["terminal_mode"]:
            raise ValueError(f"source record mismatch {rid}")
        images = [record["initial_image"]] + [turn["image"]
                                               for turn in record["turns"]]
        if not all((record_root / path).is_file() for path in images):
            raise ValueError(f"missing selected RGB frame {rid}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        prior = label["distance_to_goal_m_for_fit_label_only"]
        if len(distances) != len(prior) or \
                not all(math.isfinite(value) and value >= 0
                        for value in distances) or \
                [turn["original_turn_index"] for turn in record["turns"]] != \
                label["turn_indices"]:
            raise ValueError(f"turn alignment mismatch {rid}")
        drift = max(abs(a - b) for a, b in zip(distances, prior))
        max_turn_drift = max(max_turn_drift, drift)
        if drift > .25 or \
                record["progress_turns_geodesic_1m"] != label["forward_turns_1m"] or \
                record["regression_turns_geodesic_1m"] != label["regression_turns_1m"] or \
                abs(record["replayed_terminal_distance_m_for_audit_only"] -
                    plan["terminal_distance_m_for_replay_audit_only"]) > .25:
            raise ValueError(f"selected label or terminal drift {rid}")
        counts["forward_turns_1m"] += label["forward_turns_1m"]
        counts["regression_turns_1m"] += label["regression_turns_1m"]
        counts["images"] += len(images)
        per_episode[eid] += 1
    if len(selected["selected"]["fit"]) != 512 or \
            set(per_episode) != expected_ids or \
            any(count != 2 for count in per_episode.values()) or \
            counts["forward_turns_1m"] != summary["progress_turns_geodesic_1m"] or \
            counts["regression_turns_1m"] != summary["regression_turns_geodesic_1m"] or \
            counts["regression_turns_1m"] != selection["chosen"]["regression_turns_1m"]:
        raise ValueError("selected RGB replay summary mismatch")
    rgb_seconds = float(args.rgb_seconds_file.read_text().strip())
    if not math.isfinite(rgb_seconds) or rgb_seconds < 0:
        raise ValueError("invalid RGB replay runtime")
    report = {
        "schema": "control_fit_render_audit_v1",
        "interpretation": "R2R-train fit-only label/RGB coverage, no learned reward or navigation result.",
        "source_sha256": {"ids": EXPECTED_IDS_SHA,
                          "all_variant_manifest": source_sha,
                          "render_manifest": selected_sha,
                          "selection_report": digest(args.selection_report)},
        "selected_trajectories": 512, "unique_episode_ids": 256,
        "counts": dict(counts),
        "max_abs_turn_distance_drift_m": max_turn_drift,
        "label_only_replay_seconds": label_summary["elapsed_seconds"],
        "rgb_replay_seconds": rgb_seconds,
        "passed_sample_gate": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
