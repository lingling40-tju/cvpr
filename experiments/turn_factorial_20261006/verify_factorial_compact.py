"""Independently recount the frozen four-arm val-seen development screen.

Input is a compact per-episode export, not the remote paired analyzer's summary.
The manifest hash, arm names, threshold, and bootstrap procedure are fixed here
before reading any factorial evaluation outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

MANIFEST_SHA256 = "c3c11ac6db4e3f040be67bb6cfe58de73ab159cf5aa7251f278ac218a0ec0325"
CONTROL = "qwen3_exact_control_64step_seed11"
CANDIDATES = (
    "turn_rloo_terminal_64step_seed11",
    "dense_grpo_64step_seed11",
    "turn_rloo_64step_seed11",
)
ARMS = (CONTROL,) + CANDIDATES


def check_result(result: dict, episode_id: str, arm: str) -> tuple[int, float]:
    # Raw stats identify the episode by filename and may omit this field;
    # the exporter serializes an omitted field as null. Row/manifest order
    # is checked separately below, and a non-null declared ID must match.
    declared_id = result.get("episode_id")
    if declared_id is not None and str(declared_id) != episode_id:
        raise ValueError(f"episode mismatch: {arm} {episode_id}")
    if result.get("early_stop_reason") == "inference_error":
        raise ValueError(f"inference error: {arm} {episode_id}")
    success = result.get("success")
    spl = result.get("spl")
    if isinstance(success, bool):
        success = int(success)
    if success not in (0, 0.0, 1, 1.0):
        raise ValueError(f"invalid success: {arm} {episode_id}")
    if isinstance(spl, bool) or not isinstance(spl, (int, float)) or not math.isfinite(spl) or not 0 <= spl <= 1:
        raise ValueError(f"invalid SPL: {arm} {episode_id}")
    return int(success), float(spl)


def percentile_interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[249], ordered[9749]]  # 2.5% and 97.5% of 10,000 draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("compact", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    compact_bytes = args.compact.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest SHA-256 mismatch")
    compact = json.loads(compact_bytes)
    manifest = json.loads(manifest_bytes)
    if compact.get("schema") != "turn_factorial_val_seen256_compact_v1":
        raise ValueError("unexpected compact schema")
    if compact.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("compact references another manifest")
    if compact.get("split") != manifest.get("split") or manifest.get("split") != "val_seen":
        raise ValueError("unexpected split")
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    rows = compact["rows"]
    if len(ids) != len(set(ids)) or len(ids) != len(rows) or len(ids) != len(scenes) or len(ids) != 256:
        raise ValueError("incomplete or duplicate episode coverage")
    if len(set(scenes)) != 50:
        raise ValueError("unexpected scene coverage")
    if [str(r["episode_id"]) for r in rows] != ids:
        raise ValueError("compact episode order differs from frozen manifest")

    successes = {arm: 0 for arm in ARMS}
    spl_sums = {arm: 0.0 for arm in ARMS}
    by_scene = {name: defaultdict(list) for name in CANDIDATES}
    discordance = {name: [0, 0] for name in CANDIDATES}
    for row, episode_id, scene in zip(rows, ids, scenes):
        if str(row["scene_id"]) != scene or set(row) != {"episode_id", "scene_id", *ARMS}:
            raise ValueError(f"row fields or scene mismatch: {episode_id}")
        metrics = {}
        for arm in ARMS:
            metrics[arm] = check_result(row[arm], episode_id, arm)
            successes[arm] += metrics[arm][0]
            spl_sums[arm] += metrics[arm][1]
        c_success, c_spl = metrics[CONTROL]
        for name in CANDIDATES:
            t_success, t_spl = metrics[name]
            by_scene[name][scene].append((t_success - c_success, t_spl - c_spl))
            discordance[name][0] += int(t_success == 1 and c_success == 0)
            discordance[name][1] += int(t_success == 0 and c_success == 1)

    comparisons = {}
    for name in CANDIDATES:
        sr_delta = 100 * (successes[name] - successes[CONTROL]) / len(rows)
        spl_delta = 100 * (spl_sums[name] - spl_sums[CONTROL]) / len(rows)
        if discordance[name][0] - discordance[name][1] != successes[name] - successes[CONTROL]:
            raise ValueError(f"discordance recount mismatch: {name}")
        rng = random.Random(2026100600 + CANDIDATES.index(name))
        scene_names = sorted(by_scene[name])
        draws = []
        for _ in range(10000):
            sampled = rng.choices(scene_names, k=len(scene_names))
            values = [x for scene in sampled for x in by_scene[name][scene]]
            draws.append((100 * sum(x[0] for x in values) / len(values),
                          100 * sum(x[1] for x in values) / len(values)))
        comparisons[name] = {
            "successes": successes[name],
            "sr": successes[name] / len(rows),
            "spl": spl_sums[name] / len(rows),
            "paired_sr_points": sr_delta,
            "paired_spl_points": spl_delta,
            "candidate_only_success": discordance[name][0],
            "control_only_success": discordance[name][1],
            "scene_cluster_bootstrap_95pct_sr_points": percentile_interval([x[0] for x in draws]),
            "scene_cluster_bootstrap_95pct_spl_points": percentile_interval([x[1] for x in draws]),
            "advance_gate_passed": sr_delta >= 2.0 and spl_delta >= 2.0,
        }
    output = {
        "schema": "turn_factorial_val_seen256_independent_recount_v1",
        "split": "val_seen",
        "training_seed": 11,
        "episodes": len(rows),
        "scenes": len(set(scenes)),
        "manifest_sha256": MANIFEST_SHA256,
        "compact_sha256": hashlib.sha256(compact_bytes).hexdigest(),
        "control": CONTROL,
        "control_successes": successes[CONTROL],
        "control_sr": successes[CONTROL] / len(rows),
        "control_spl": spl_sums[CONTROL] / len(rows),
        "comparisons": comparisons,
        "inference_errors": 0,
        "interpretation": "One-seed val-seen development comparison; scene-cluster intervals are descriptive only.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
