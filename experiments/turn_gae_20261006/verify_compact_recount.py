"""Recount the exported GAE development comparison without remote raw logs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent / "real64" / "val_seen778"
CONTROL = "qwen3_exact_control_64step_seed11"
CANDIDATE = "turn_gae_64step_seed11"
EXPECTED_MANIFEST_SHA = "03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed"


def main() -> None:
    manifest_bytes = (ROOT / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED_MANIFEST_SHA
    manifest = json.loads(manifest_bytes)
    compact_bytes = (ROOT / "compact.json").read_bytes()
    compact = json.loads(compact_bytes)
    report = json.loads((ROOT / "independent_recount.json").read_text())
    assert report["compact_sha256"] == hashlib.sha256(compact_bytes).hexdigest()
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    rows = compact["rows"]
    assert len(ids) == len(set(ids)) == len(rows) == 778
    assert len(set(scenes)) == 53
    assert [r["episode_id"] for r in rows] == ids
    assert [r["scene_id"] for r in rows] == scenes
    success = {CONTROL: 0, CANDIDATE: 0}
    spl = {CONTROL: 0.0, CANDIDATE: 0.0}
    candidate_only = control_only = 0
    path_length = {CONTROL: 0.0, CANDIDATE: 0.0}
    for row in rows:
        for label in (CONTROL, CANDIDATE):
            arm = row[label]
            assert arm["early_stop_reason"] != "inference_error"
            assert arm["success"] in (False, True, 0, 1, 0.0, 1.0)
            assert math.isfinite(float(arm["spl"])) and 0 <= float(arm["spl"]) <= 1
            assert math.isfinite(float(arm["path_length"])) and float(arm["path_length"]) >= 0
            success[label] += int(bool(arm["success"]))
            spl[label] += float(arm["spl"])
            path_length[label] += float(arm["path_length"])
        candidate_only += int(bool(row[CANDIDATE]["success"]) and not bool(row[CONTROL]["success"]))
        control_only += int(bool(row[CONTROL]["success"]) and not bool(row[CANDIDATE]["success"]))
    n = len(rows)
    summary = {
        "schema": "gae_val_seen778_local_compact_recount_v1",
        "episodes": n,
        "scenes": len(set(scenes)),
        "control_successes": success[CONTROL],
        "candidate_successes": success[CANDIDATE],
        "control_spl": spl[CONTROL] / n,
        "candidate_spl": spl[CANDIDATE] / n,
        "paired_sr_points": 100 * (success[CANDIDATE] - success[CONTROL]) / n,
        "paired_spl_points": 100 * (spl[CANDIDATE] - spl[CONTROL]) / n,
        "candidate_only_success": candidate_only,
        "control_only_success": control_only,
        "control_mean_path_length_m": path_length[CONTROL] / n,
        "candidate_mean_path_length_m": path_length[CANDIDATE] / n,
        "inference_errors": 0,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "compact_sha256": hashlib.sha256(compact_bytes).hexdigest(),
    }
    for key in ("episodes", "scenes", "control_successes", "candidate_successes",
                "candidate_only_success", "control_only_success", "inference_errors", "manifest_sha256"):
        assert summary[key] == report[key], key
    for key in ("control_spl", "candidate_spl", "paired_sr_points", "paired_spl_points"):
        assert math.isclose(summary[key], report[key], abs_tol=1e-9), key
    (ROOT / "local_compact_recount.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
