"""Independently recount the fixed val-seen paired navigation screen."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

CONTROL = "qwen3_exact_control_64step_seed11"
CANDIDATE = "turn_rloo_64step_seed11"
MANIFEST_SHA256 = "d9ba66de3fb4fc070cae9449decb1427e80672d61523891af6f8d8ad45ad6a31"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("compact", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    compact_bytes = args.compact.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256
    source = json.loads(compact_bytes)
    manifest = json.loads(manifest_bytes)
    assert source["schema"] == "turn_rloo_val_seen256_compact_v1"
    assert source["manifest_sha256"] == MANIFEST_SHA256
    assert source["split"] == manifest["split"] == "val_seen"
    expected = list(map(str, manifest["episode_ids"]))
    rows = source["rows"]
    assert len(expected) == len(set(expected)) == len(rows) == 256
    assert len(manifest["scene_ids"]) == 256
    assert [r["episode_id"] for r in rows] == expected
    scenes = defaultdict(list)
    successes = {CONTROL: 0, CANDIDATE: 0}
    spl_totals = {CONTROL: 0.0, CANDIDATE: 0.0}
    candidate_only = control_only = 0
    for row, scene in zip(rows, manifest["scene_ids"]):
        assert row["scene_id"] == str(scene)
        for arm in (CONTROL, CANDIDATE):
            result = row[arm]
            success = result["success"]
            spl = result["spl"]
            assert success in (0, 0.0, 1, 1.0)
            assert isinstance(spl, (int, float)) and 0 <= spl <= 1
            assert result["early_stop_reason"] != "inference_error"
            successes[arm] += int(success)
            spl_totals[arm] += spl
        c = int(row[CONTROL]["success"])
        t = int(row[CANDIDATE]["success"])
        candidate_only += int(t == 1 and c == 0)
        control_only += int(c == 1 and t == 0)
        scenes[str(scene)].append((t - c, row[CANDIDATE]["spl"] - row[CONTROL]["spl"]))
    assert len(scenes) == 53
    assert candidate_only - control_only == successes[CANDIDATE] - successes[CONTROL]
    sr_delta = 100 * (successes[CANDIDATE] - successes[CONTROL]) / len(rows)
    spl_delta = 100 * (spl_totals[CANDIDATE] - spl_totals[CONTROL]) / len(rows)
    # Independent descriptive scene-cluster bootstrap, with a different seed
    # and number of resamples from the remote analysis.
    rng = random.Random(48151623)
    scene_names = sorted(scenes)
    boots = []
    for _ in range(10000):
        selected = rng.choices(scene_names, k=len(scene_names))
        total_n = total_sr = total_spl = 0
        for scene in selected:
            entries = scenes[scene]
            total_n += len(entries)
            total_sr += sum(x[0] for x in entries)
            total_spl += sum(x[1] for x in entries)
        boots.append((100 * total_sr / total_n, 100 * total_spl / total_n))
    def bounds(index):
        ordered = sorted(x[index] for x in boots)
        return [ordered[249], ordered[9749]]
    output = {
        "schema": "turn_rloo_val_seen256_independent_recount_v1",
        "split": "val_seen",
        "training_seed": 11,
        "episodes": len(rows),
        "scenes": len(scenes),
        "manifest_sha256": MANIFEST_SHA256,
        "compact_sha256": hashlib.sha256(compact_bytes).hexdigest(),
        "control_successes": successes[CONTROL],
        "candidate_successes": successes[CANDIDATE],
        "control_sr": successes[CONTROL] / len(rows),
        "candidate_sr": successes[CANDIDATE] / len(rows),
        "paired_sr_points": sr_delta,
        "control_spl": spl_totals[CONTROL] / len(rows),
        "candidate_spl": spl_totals[CANDIDATE] / len(rows),
        "paired_spl_points": spl_delta,
        "candidate_only_success": candidate_only,
        "control_only_success": control_only,
        "inference_errors": 0,
        "scene_cluster_bootstrap": {
            "resamples": len(boots),
            "seed": 48151623,
            "descriptive_sr_95pct_points": bounds(0),
            "descriptive_spl_95pct_points": bounds(1),
        },
        "interpretation": "One seed, one stochastic decode, val-seen development set; descriptive interval only.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
