"""Independent exact-cover, JPEG, source-distance and reused-byte audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_part(part: str, manifest: dict, labels: dict, report: dict,
                old: dict, rgb: Path, old_rgb: Path, manifest_sha: str,
                shards: int, target: str) -> dict:
    all_plans = manifest["selected"][part]
    plans = [x for x in all_plans if not target or x["record_id"] == target]
    if not plans or len({x["record_id"] for x in plans}) != len(plans):
        raise ValueError("empty or repeated selected records")
    selected = {x["record_id"]: x for x in plans}
    old_by_id = {x["record_id"]: x for x in old["selected"][part]}
    old_paths = {}
    for rid, plan in old_by_id.items():
        if rid not in selected:
            continue
        record = read(old_rgb / part / "records" / f"{rid}.json")
        if record["manifest_sha256"] != manifest[
                "boundary_manifest_sha256"]:
            raise ValueError("changed prior verified RGB")
        for state, role in ((plan["outside_state_index"], "outside"),
                            (plan["inside_state_index"], "inside")):
            old_paths[(rid, str(state))] = old_rgb / part / record[
                "input"]["images"][role]
    summaries = [read(rgb / part / ("summary.json" if shards == 1 else
                 f"summary.shard{i}.json")) for i in range(shards)]
    if any(s["capture_manifest_sha256"] != manifest_sha or
           s["part"] != part or s["shards"] != shards or s["shard"] != i or
           s["requested"] != s["completed"] or s["errors"]
           for i, s in enumerate(summaries)) or \
            sum(s["requested"] for s in summaries) != len(plans):
        raise ValueError(f"incomplete replay summaries: {part}")
    max_drift = 0.0
    copied = 0
    state_count = 0
    for rid, plan in selected.items():
        rec = read(rgb / part / "records" / f"{rid}.json")
        audit = read(rgb / part / "audits" / f"{rid}.json")
        if rec["schema"] != "multiview_event_rgb_input_v1" or \
                audit["schema"] != "multiview_event_rgb_replay_audit_v1" or \
                rec["capture_manifest_sha256"] != manifest_sha or \
                audit["capture_manifest_sha256"] != manifest_sha or \
                rec["record_id"] != audit["record_id"] != rid or \
                (audit["seed"], str(audit["episode_id"]),
                 audit["variant"], audit["scene_id"]) != \
                (plan["seed"], str(plan["episode_id"]),
                 plan["variant"], plan["scene_id"]):
            raise ValueError(f"replay identity mismatch: {rid}")
        if plan["scene_id"] not in manifest["scene_split"][part]:
            raise ValueError(f"scene partition mismatch: {rid}")
        inp = rec["input"]
        if set(inp) != {"instruction", "terminal_clause",
                        "wrong_instruction", "wrong_terminal_clause",
                        "images_by_state"} or \
                any(inp[key] != plan[key] for key in (
                    "instruction", "terminal_clause", "wrong_instruction",
                    "wrong_terminal_clause")) or \
                set(inp["images_by_state"]) != \
                {str(x) for x in plan["states_to_capture"]} or \
                set(audit["state_distance_m"]) != \
                set(inp["images_by_state"]) or \
                set(audit["source_state_distance_m"]) != \
                set(inp["images_by_state"]):
            raise ValueError(f"model input or state mismatch: {rid}")
        if any(bad in json.dumps(inp).lower() for bad in
               ("oracle_distance", "distance_to_goal", "task_success")):
            raise ValueError(f"privileged field in model input: {rid}")
        per_record_copied = 0
        for state, rel in inp["images_by_state"].items():
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or \
                    path.parts[:2] != ("frames", rid) or \
                    path.name != f"{int(state):04d}.jpg":
                raise ValueError(f"unsafe RGB path: {rid}")
            full = rgb / part / path
            with Image.open(full) as image:
                if image.format != "JPEG" or \
                        not 1 <= image.width <= 336 or \
                        not 1 <= image.height <= 336:
                    raise ValueError(f"invalid replay JPEG: {rid}")
                image.verify()
            old_path = old_paths.get((rid, state))
            if old_path:
                if digest(full) != digest(old_path):
                    raise ValueError(f"reused frame changed bytes: {rid}")
                per_record_copied += 1
            delta = abs(float(audit["state_distance_m"][state]) - float(
                audit["source_state_distance_m"][state]))
            if not math.isfinite(delta) or delta > .25:
                raise ValueError(f"state geodesic drift: {rid} {delta}")
            max_drift = max(max_drift, delta)
            state_count += 1
        terminal = abs(float(audit["source_terminal_distance_m"])-float(
            audit["replay_terminal_distance_m"]))
        if not math.isfinite(terminal) or terminal > .25 or \
                audit["copied_verified_crossing_images"] != per_record_copied:
            raise ValueError(f"terminal or copied-frame mismatch: {rid}")
        max_drift = max(max_drift, terminal)
        copied += per_record_copied
    for label in labels["selected"][part]:
        rid = label["record_id"]
        if rid not in selected:
            continue
        pair_id = rid + ":" + label["kind"]
        if label["pair_id"] != pair_id or label["kind"] not in (
                "crossing", "far_nonarrival", "wrong_instruction",
                "retreat"):
            raise ValueError(f"changed pair class: {pair_id}")
        audit = read(rgb / part / "audits" / f"{rid}.json")
        for role in ("before", "after"):
            state = str(label[f"{role}_state_index"])
            if state not in audit["state_distance_m"] or \
                    abs(float(audit["state_distance_m"][state]) - float(
                        label[f"{role}_distance_m_for_audit_only"])) > .25:
                raise ValueError(f"pair geodesic mismatch: {pair_id}")
    if not target and (len(plans) != report["capture_records"][part] or
                       state_count != report["capture_state_count"][part] or
                       copied != report["verified_crossing_states_reusable"][part] or
                       state_count-copied != report["extra_states_to_render"][part]):
        raise ValueError(f"incomplete source state counts: {part}")
    return {"part": part, "records": len(plans),
            "state_images": state_count,
            "reused_verified_images": copied,
            "new_rendered_images": state_count-copied,
            "max_geodesic_drift_m": max_drift,
            "shards": shards}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("capture-manifest", "pair-labels", "source-report",
                 "boundary-manifest", "rgb-root", "old-rgb-root",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--smoke-record-id", default="")
    args = parser.parse_args()
    manifest, labels, report, old = map(
        read, (args.capture_manifest, args.pair_labels,
               args.source_report, args.boundary_manifest))
    manifest_sha = digest(args.capture_manifest)
    if manifest["schema"] != "multiview_event_rgb_capture_manifest_v1" or \
            labels["schema"] != "multiview_event_privileged_pair_labels_v1" or \
            report["schema"] != "multiview_event_frozen_source_report_v1" or \
            labels["capture_manifest_sha256"] != manifest_sha or \
            report["capture_manifest_sha256"] != manifest_sha or \
            report["labels_sha256"] != digest(args.pair_labels) or \
            manifest["boundary_manifest_sha256"] != \
                digest(args.boundary_manifest) or \
            report["boundary_manifest_sha256"] != \
                digest(args.boundary_manifest) or \
            not report["ready_for_rgb_replay"] or report["audit_selected"] or \
            not 1 <= args.shards <= 4:
        raise ValueError("changed frozen n4 source")
    parts = ("fit",) if args.smoke_record_id else ("fit", "development")
    value = {"schema": "multiview_event_rgb_verification_v1",
             "capture_manifest_sha256": manifest_sha,
             "pair_labels_sha256": digest(args.pair_labels),
             "smoke_record_id": args.smoke_record_id or None,
             "parts": [verify_part(part, manifest, labels, report, old,
                                   args.rgb_root, args.old_rgb_root,
                                   manifest_sha, args.shards,
                                   args.smoke_record_id)
                       for part in parts],
             "audit_opened": False, "navigation_result": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
