"""Fail-closed, CPU-only audit of gated sparse RGB replay before LoRA fit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from PIL import Image

from preflight_group_future_advantage_pool import coverage_checks
from prepare_future_advantage_sparse_manifest import digest, required_capture_turns


def _record_path(part_root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("sparse RGB path escapes part root")
    full = part_root / path
    if not full.is_file():
        raise ValueError(f"missing sparse RGB frame: {relative}")
    return full


def _finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite replay distance")
    return number


def verify(manifest_path: Path, labels_path: Path, report_path: Path,
           replay_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(labels_path.read_text())
    report = json.loads(report_path.read_text())
    if report.get("schema") != "group_future_advantage_exact512_pooled_preflight_v1" or \
            report.get("coverage_checks") != coverage_checks(report) or \
            report.get("enough_coverage_for_fit_preparation") is not True or \
            not all(report["coverage_checks"].values()):
        raise ValueError("frozen n=4 pooled coverage gate did not pass")
    if manifest.get("schema") != "future_advantage_sparse_replay_manifest_v1" or \
            labels.get("schema") != "future_advantage_within_group_pair_labels_v1" or \
            manifest.get("group_size") != 4 or labels.get("group_size") != 4 or \
            manifest.get("anchors") != [3, 6] or \
            labels.get("anchors") != [3, 6] or \
            manifest.get("seeds") != labels.get("seeds") or \
            manifest.get("seeds") != report.get("seeds") or \
            manifest.get("source_sha256") != report.get("source_sha256") or \
            manifest.get("preflight_report_sha256") != digest(report_path) or \
            labels.get("preflight_report_sha256") != digest(report_path) or \
            labels.get("replay_manifest_sha256") != digest(manifest_path) or \
            manifest.get("counts") != labels.get("counts"):
        raise ValueError("sparse replay sources do not match passed gate")
    output = {"schema": "future_advantage_sparse_replay_verification_v1",
              "group_size": 4, "seeds": manifest["seeds"],
              "source_sha256": {"manifest": digest(manifest_path),
                                "labels": digest(labels_path),
                                "preflight": digest(report_path)},
              "parts": {}}
    scenes = {}
    for part in ("fit", "development"):
        part_root = replay_root / part
        selected = manifest["selected"][part]
        expected_ids = {plan["record_id"] for plan in selected}
        if len(expected_ids) != len(selected) or not selected:
            raise ValueError(f"duplicate or empty {part} replay selection")
        actual_records = {path.stem for path in (part_root / "records").glob("*.json")}
        actual_audits = {path.stem for path in (part_root / "audits").glob("*.json")}
        if actual_records != expected_ids or actual_audits != expected_ids:
            raise ValueError(f"incomplete or extra sparse records in {part}")
        scene_ids, episode_ids = set(), set()
        frames, bytes_total, max_drift = 0, 0, 0.0
        by_id = {}
        for plan in selected:
            rid = plan["record_id"]
            by_id[rid] = plan
            scene_ids.add(plan["scene_id"])
            episode_ids.add(str(plan["episode_id"]))
            record = json.loads((part_root / "records" / f"{rid}.json").read_text())
            audit = json.loads((part_root / "audits" / f"{rid}.json").read_text())
            if set(record) != {"schema", "manifest_sha256", "record_id", "input"} or \
                    record["schema"] != "future_advantage_sparse_model_input_v1" or \
                    record["record_id"] != rid or \
                    record["manifest_sha256"] != output["source_sha256"]["manifest"] or \
                    set(record["input"]) != {"instruction", "images",
                                              "action_history_by_anchor"} or \
                    record["input"]["instruction"] != plan["instruction"]:
                raise ValueError(f"privileged, incomplete, or changed model input: {rid}")
            anchors = required_capture_turns(plan["anchor_turns"])
            if anchors != plan["anchor_turns"]:
                raise ValueError(f"turn-6 intermediate image missing in plan: {rid}")
            images = record["input"]["images"]
            histories = record["input"]["action_history_by_anchor"]
            if set(images) != {"0"} | {str(a) for a in anchors} or \
                    set(histories) != {str(a) for a in anchors}:
                raise ValueError(f"missing or future sparse prefix fields: {rid}")
            for anchor in anchors:
                history = histories[str(anchor)]
                if len(history) != anchor or [row["turn"] for row in history] != \
                        list(range(1, anchor + 1)) or any(
                            not row["executed_actions"] or any(
                                "stop" in action.lower()
                                for action in row["executed_actions"])
                            for row in history):
                    raise ValueError(f"invalid executed history: {rid}/{anchor}")
            for image in images.values():
                path = _record_path(part_root, image)
                with Image.open(path) as frame:
                    frame.verify()
                frames += 1
                bytes_total += path.stat().st_size
            terminal = _finite(audit["terminal_distance_m"])
            source_terminal = _finite(plan["terminal_distance_m_for_replay_audit_only"])
            if audit.get("schema") != "future_advantage_sparse_replay_audit_v1" or \
                    audit.get("record_id") != rid or \
                    audit.get("manifest_sha256") != output["source_sha256"]["manifest"] or \
                    audit.get("seed") != plan["seed"] or \
                    str(audit.get("episode_id")) != str(plan["episode_id"]) or \
                    audit.get("variant") != plan["variant"] or \
                    audit.get("scene_id") != plan["scene_id"] or \
                    abs(terminal - source_terminal) > .25 or \
                    abs(_finite(audit["source_terminal_distance_m"]) -
                        source_terminal) > 1e-6 or \
                    abs(_finite(audit["terminal_drift_m"]) -
                        (terminal - source_terminal)) > 1e-6:
                raise ValueError(f"replay audit drift or identity mismatch: {rid}")
            max_drift = max(max_drift, abs(terminal - source_terminal))
        if frames != sum(1 + len(p["anchor_turns"]) for p in selected):
            raise ValueError(f"sparse frame count mismatch: {part}")
        counts = defaultdict(int)
        for pair in labels["pairs"][part]:
            anchor, seed, eid = pair["anchor"], pair["seed"], str(pair["episode_id"])
            left = f"s{seed}_e{eid}_v{pair['left_variant']}"
            right = f"s{seed}_e{eid}_v{pair['right_variant']}"
            if anchor not in (3, 6) or left not in by_id or right not in by_id or \
                    anchor not in by_id[left]["anchor_turns"] or \
                    anchor not in by_id[right]["anchor_turns"]:
                raise ValueError(f"pair lacks audited RGB prefix: {part}")
            counts[str(anchor)] += 1
        for anchor in (3, 6):
            key = str(anchor)
            expected = manifest["counts"][part][key]
            broad = report["parts"][part]["anchors"][key]
            same = report["parts"][part]["same_terminal_mode_anchors"][key]
            if counts[key] != expected["pairs"] or \
                    expected["pairs"] != broad["pairs"] or \
                    expected["unique_episode_groups"] != broad["episode_groups"] or \
                    expected["same_terminal_mode_pairs"] != same["pairs"] or \
                    expected["same_terminal_mode_unique_episode_groups"] != \
                    same["episode_groups"]:
                raise ValueError(f"pair/manifest count mismatch: {part}/{anchor}")
        scenes[part] = scene_ids
        output["parts"][part] = {"records": len(selected),
                                  "unique_episode_ids": len(episode_ids),
                                  "scenes": len(scene_ids),
                                  "frames": frames, "frame_bytes": bytes_total,
                                  "max_terminal_drift_m": max_drift,
                                  "pairs_by_anchor": dict(counts)}
    if scenes["fit"] & scenes["development"]:
        raise ValueError("fit/development scenes overlap")
    output["complete"] = True
    output["interpretation"] = (
        "Source and replay integrity only; not an observation-reward or "
        "navigation-performance result")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "report", "replay-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest, args.labels, args.report, args.replay_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
