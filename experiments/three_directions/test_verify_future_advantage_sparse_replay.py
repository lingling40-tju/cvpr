"""Exercise sparse RGB verification with a tiny synthetic gated fixture."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from PIL import Image

from test_future_advantage_sparse_lora import fixture, write
from verify_future_advantage_sparse_replay import digest, verify


def rejects(call, description: str) -> None:
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError(f"verifier accepted {description}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sparse_rgb_verify_") as tmp:
        root = Path(tmp)
        manifest_path, labels_path, replay = fixture(root)
        manifest = json.loads(manifest_path.read_text())
        labels = json.loads(labels_path.read_text())
        manifest["source_sha256"] = {"synthetic": True}
        report = {"schema": "group_future_advantage_exact512_pooled_preflight_v1",
                  "seeds": [11], "source_sha256": manifest["source_sha256"],
                  "coverage_checks": {},
                  "enough_coverage_for_fit_preparation": True,
                  "parts": {}}
        for part in ("fit", "development"):
            report["parts"][part] = {
                "anchors": {str(a): {"pairs": 1, "episode_groups": 1}
                            for a in (3, 6)},
                "same_terminal_mode_anchors": {
                    str(a): {"pairs": 1, "episode_groups": 1}
                    for a in (3, 6)}}
            for plan in manifest["selected"][part]:
                plan.update({"instruction": "Walk toward the chair.",
                             "anchor_turns": [3, 6],
                             "terminal_distance_m_for_replay_audit_only": 2.0})
        report_path = root / "report.json"
        write(report_path, report)
        manifest["preflight_report_sha256"] = digest(report_path)
        write(manifest_path, manifest)
        labels["preflight_report_sha256"] = digest(report_path)
        labels["replay_manifest_sha256"] = digest(manifest_path)
        write(labels_path, labels)
        first_record = None
        for part in ("fit", "development"):
            for plan in manifest["selected"][part]:
                rid = plan["record_id"]
                part_root = replay / part
                image_paths = {}
                for turn in (0, 3, 6):
                    path = part_root / "frames" / rid / f"{turn:04d}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), (turn * 20, 40, 60)).save(path)
                    image_paths[str(turn)] = str(path.relative_to(part_root))
                history = [{"turn": t, "executed_actions": [
                    "move forward 25cm"]} for t in range(1, 7)]
                record_path = part_root / "records" / f"{rid}.json"
                write(record_path, {
                    "schema": "future_advantage_sparse_model_input_v1",
                    "record_id": rid, "manifest_sha256": digest(manifest_path),
                    "input": {"instruction": plan["instruction"],
                              "images": image_paths,
                              "action_history_by_anchor": {
                                  "3": history[:3], "6": history}}})
                if first_record is None:
                    first_record = record_path
                write(part_root / "audits" / f"{rid}.json", {
                    "schema": "future_advantage_sparse_replay_audit_v1",
                    "record_id": rid, "manifest_sha256": digest(manifest_path),
                    "seed": plan["seed"], "episode_id": plan["episode_id"],
                    "variant": plan["variant"], "scene_id": plan["scene_id"],
                    "terminal_distance_m": 2.1,
                    "source_terminal_distance_m": 2.0,
                    "terminal_drift_m": .1})
        # The small fixture stands in for the upstream coverage gate. The
        # production verifier still recomputes the real frozen thresholds.
        with patch("verify_future_advantage_sparse_replay.coverage_checks",
                   return_value={}):
            result = verify(manifest_path, labels_path, report_path, replay)
            assert result["complete"]
            assert result["parts"]["fit"]["frames"] == 6
            assert result["parts"]["development"]["frames"] == 6
            record = json.loads(first_record.read_text())
            record["input"]["privileged_goal_distance"] = 2.0
            write(first_record, record)
            rejects(lambda: verify(manifest_path, labels_path, report_path,
                                   replay), "privileged model input")
            del record["input"]["privileged_goal_distance"]
            write(first_record, record)
            (replay / "fit" / record["input"]["images"]["3"]).unlink()
            rejects(lambda: verify(manifest_path, labels_path, report_path,
                                   replay), "missing turn-3 context image")
    print("sparse replay verification checks passed")


if __name__ == "__main__":
    main()
