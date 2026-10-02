"""Independently recompute the archived fixed-256 progress pilot result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


root = Path(__file__).resolve().parent
archive = root / "progress64"
manifest_path = root / "val256" / "manifest.json"
manifest = json.loads(manifest_path.read_text())
analysis = json.loads((archive / "paired_progress64_vs_branch_control64.json").read_text())
validation = json.loads((archive / "progress64_seed11.validated.json").read_text())
audit = json.loads((archive / "paired_train_audit.json").read_text())
training = json.loads((archive / "validation.json").read_text())
rows = [json.loads(line) for line in
        (archive / "paired_progress64_vs_branch_control64.jsonl").read_text().splitlines()]

ids = [str(x) for x in manifest["episode_ids"]]
scenes = [str(x) for x in manifest["scene_ids"]]
assert len(ids) == len(set(ids)) == len(rows) == analysis["episodes"] == 256
assert len(set(scenes)) == analysis["scenes"] == 11
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == analysis["manifest_sha256"]
assert [(r["episode_id"], r["scene_id"]) for r in rows] == list(zip(ids, scenes))
assert analysis["candidate"] == validation["label"] == "progress64_seed11"
assert analysis["control"] == "branch_control64"
assert validation["episodes"] == 256 and validation["inference_errors"] == 0

for prefix, key in (("progress", "candidate_metrics"), ("control", "control_metrics")):
    metrics = analysis[key]
    assert metrics["count"] == 256 and metrics["inference_errors"] == 0
    assert sum(bool(r[prefix + "_success"]) for r in rows) == metrics["successes"]
    for field, column in (("sr", prefix + "_success"),
                          ("spl", prefix + "_spl"),
                          ("mean_distance_to_goal", prefix + "_distance_to_goal")):
        observed = sum(float(r[column]) for r in rows) / 256
        assert math.isclose(observed, metrics[field], abs_tol=1e-12)

paired = analysis["paired"]
sr_pp = 100 * sum(int(r["progress_success"]) - int(r["control_success"])
                  for r in rows) / 256
spl_pp = 100 * sum(r["progress_spl"] - r["control_spl"] for r in rows) / 256
assert math.isclose(sr_pp, paired["sr_pp"], abs_tol=1e-12)
assert math.isclose(spl_pp, paired["spl_pp"], abs_tol=1e-12)
assert paired["candidate_only_successes"] == sum(
    r["progress_success"] and not r["control_success"] for r in rows)
assert paired["control_only_successes"] == sum(
    r["control_success"] and not r["progress_success"] for r in rows)
assert audit["matched_train_episode_sets_at_each_step"] == training["steps"] == 64
assert audit["progress"]["rollouts"] == 512
assert audit["progress"]["progress_components_seen"] == 512
assert audit["progress"]["nonzero_progress_rollouts"] == 510
assert audit["progress"]["invalid_progress_distances"] == 0
assert len(training["actor_grad_norms"]) == 64
assert all(math.isfinite(x) and x > 0 for x in training["actor_grad_norms"])

print(f"verified 256 paired episodes; progress-control: SR {sr_pp:+.6f} pp, "
      f"SPL {spl_pp:+.6f} pp")
