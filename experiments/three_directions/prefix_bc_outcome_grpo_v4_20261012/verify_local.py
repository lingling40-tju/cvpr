#!/usr/bin/env python3
"""Independent arithmetic, coverage, and identity check for the prefix-BC v4 package."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(rel: str):
    return json.loads((ROOT / rel).read_text())

def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()

identity = read("runlogs/freeze/protocol_identity_v4.json")
checked = []
for rel, item in identity["files"].items():
    path = ROOT / rel
    if path.is_file():
        assert path.stat().st_size == item["bytes"], rel
        assert sha(rel) == item["sha256"], rel
        checked.append(rel)
assert len(checked) >= 7

manifest_rel = "prepared_data/development256.json"
manifest = read(manifest_rel)
assert sha(manifest_rel) == "8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3"
expected_ids = [str(x) for x in manifest["episode_ids"]]
pair_rel = "runlogs/prefix_bc_development_eval_v2/paired_episodes_v2.jsonl"
rows = [json.loads(line) for line in (ROOT / pair_rel).read_text().splitlines() if line]
episode_ids = [str(row["episode_id"]) for row in rows]
assert len(rows) == 256 and len(set(episode_ids)) == 256
assert set(episode_ids) == set(expected_ids)
assert len({row["scene_id"] for row in rows}) == 8

candidate_success = sum(int(row["candidate_success"]) for row in rows)
sft_success = sum(int(row["control_success"]) for row in rows)
candidate_spl = sum(float(row["candidate_spl"]) for row in rows) / 256
sft_spl = sum(float(row["control_spl"]) for row in rows) / 256
sr_pp = 100 * (candidate_success - sft_success) / 256
spl_pp = 100 * (candidate_spl - sft_spl)
report = read("runlogs/prefix_bc_development_eval_v2/paired_report_v2.json")
raw = read("runlogs/prefix_bc_development_eval_v2/independent_raw_recount_v2.json")
assert candidate_success == 41 and sft_success == 109
assert abs(sr_pp - report["paired_sr_points"]) < 1e-10
assert abs(spl_pp - report["paired_spl_points"]) < 1e-10
assert abs(sr_pp - raw["paired"]["sr_points"]) < 1e-10
assert abs(spl_pp - raw["paired"]["spl_points"]) < 1e-10
assert raw["manifest_sha256"] == sha(manifest_rel)
assert raw["unique_episode_ids"] == 256 and raw["inference_errors"] == 0
assert report["scene_bootstrap_95pct_sr_points"][1] < 0
assert report["scene_bootstrap_95pct_spl_points"][1] < 0

audit = read("runlogs/prefix_bc_grpo_n4_seed11_v4/training_audit_v4.json")
assert audit["optimizer_updates_including_smoke"] == 64
assert audit["resumed_nonzero_actor_gradient_updates"] == 62
assert audit["resumed_finite_nonnegative_kl_updates"] == 62
assert audit["resumed_updates_with_active_bc_tokens"] == 17
assert (ROOT / "runlogs/prefix_bc_grpo_n4_seed11_v4/completed").read_text().strip() == audit["completed_marker"]
assert (ROOT / "runlogs/prefix_bc_development_eval_v2/suite.completed").read_text().strip()

print(json.dumps({
    "result": "PASS",
    "source_hashes_checked": len(checked),
    "episodes": 256,
    "scenes": 8,
    "candidate_successes": candidate_success,
    "sft_successes": sft_success,
    "paired_sr_points": sr_pp,
    "paired_spl_points": spl_pp,
    "inference_errors": 0,
    "training_updates": 64,
    "updates_with_active_bc_tokens": 17,
    "scope": "exploratory reused R2R-train development screen; not a clean generalization test"
}, indent=2))
