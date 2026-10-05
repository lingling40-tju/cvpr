"""Independently recount the compact n=4 stop-boundary pilot export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MANIFEST_SHA = "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c"


def near(a: float, b: float) -> None:
    if not math.isfinite(a) or abs(a - b) > 1e-9:
        raise ValueError(f"reported metric mismatch: {a} versus {b}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError("changed frozen manifest")
    ids = list(map(str, manifest["episode_ids"]))
    scenes = list(map(str, manifest["scene_ids"]))
    rows = [json.loads(line) for line in
            (package / "paired_boundary_episodes.jsonl").read_text().splitlines()]
    if len(ids) != len(set(ids)) or len(ids) != 256 or len(scenes) != 256 or \
            [row["episode_id"] for row in rows] != ids or \
            [str(row["scene_id"]) for row in rows] != scenes:
        raise ValueError("missing, duplicate, or reordered episode")
    analysis = json.loads((package / "paired_boundary_vs_control.json").read_text())
    if analysis["episodes"] != 256 or analysis["scenes"] != 9 or \
            analysis["manifest_sha256"] != MANIFEST_SHA or \
            analysis["candidate"] != "stop_boundary_64_seed11" or \
            analysis["control"] != "qwen3_exact_control_64_seed11_oracle_screen":
        raise ValueError("paired report identity mismatch")
    for arm, validator_file, label in (
            ("candidate", "candidate_validated.json", analysis["candidate"]),
            ("control", "reused_control_validation.json", analysis["control"])):
        validator = json.loads((package / validator_file).read_text())
        successes = sum(bool(row[arm]["success"]) for row in rows)
        spl = sum(float(row[arm]["spl"]) for row in rows) / 256
        distance = sum(float(row[arm]["distance_to_goal_m"])
                       for row in rows) / 256
        if validator != {"label": label, "episodes": 256,
                          "successes": successes, "inference_errors": 0}:
            raise ValueError(f"{arm} validator disagrees with episodes")
        metrics = analysis[f"{arm}_metrics"]
        if metrics["count"] != 256 or metrics["successes"] != successes or \
                metrics["inference_errors"] != 0 or \
                any(row[arm]["early_stop_reason"] == "inference_error"
                    for row in rows):
            raise ValueError(f"{arm} metric or inference error")
        near(metrics["sr"], successes / 256)
        near(metrics["spl"], spl)
        near(metrics["mean_distance_to_goal"], distance)
    candidate = analysis["candidate_metrics"]
    control = analysis["control_metrics"]
    paired = analysis["paired"]
    near(paired["sr_pp"], 100 * (candidate["sr"] - control["sr"]))
    near(paired["spl_pp"], 100 * (candidate["spl"] - control["spl"]))
    if paired["candidate_only_successes"] != sum(
            row["candidate"]["success"] and not row["control"]["success"]
            for row in rows) or paired["control_only_successes"] != sum(
            row["control"]["success"] and not row["candidate"]["success"]
            for row in rows):
        raise ValueError("paired success cells mismatch")
    audit = json.loads((package / "train_audit_64step.json").read_text())
    if audit["group_size"] != 4 or audit["steps"] != 64 or \
            audit["same_row_candidate_control"] is not True or \
            audit["unique_train_episodes"] != 256 or \
            audit["nonzero_actor_gradient_steps"] != 64:
        raise ValueError("wrong matched training audit")
    result = {
        "schema": "stop_boundary_pilot_compact_independent_recount_v1",
        "manifest_sha256": MANIFEST_SHA, "episodes": 256,
        "candidate_successes": candidate["successes"],
        "control_successes": control["successes"],
        "sr_pp": paired["sr_pp"], "spl_pp": paired["spl_pp"],
        "group_size": 4, "optimizer_steps": 64,
        "paired_export_sha256": hashlib.sha256(
            (package / "paired_boundary_episodes.jsonl").read_bytes()).hexdigest(),
        "interpretation": "Reused val-unseen development screen; privileged reward only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
