"""Export and independently recount the frozen normalized-RLOO screen.

This checker is fixed before the conditional experiment produces outcomes.
It reads raw per-episode stats, requires exact four-shard coverage for both
arms, and does not use the remote paired analyzer's aggregate numbers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

MANIFEST_SHA256 = "39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b"
CONTROL = "qwen3_exact_control_64step_seed11"
CANDIDATE = "norm_terminal_rloo_64step_seed11"
ARMS = (CONTROL, CANDIDATE)
FIELDS = ("episode_id", "success", "spl", "early_stop_reason", "distance_to_goal", "path_length", "oracle_success")


def metric(raw: dict, episode_id: str, arm: str) -> tuple[int, float]:
    if str(raw.get("episode_id", episode_id)) != episode_id:
        raise ValueError(f"episode mismatch: {arm} {episode_id}")
    if raw.get("early_stop_reason") == "inference_error":
        raise ValueError(f"inference error: {arm} {episode_id}")
    success = raw.get("success")
    if isinstance(success, bool):
        success = int(success)
    if success not in (0, 0.0, 1, 1.0):
        raise ValueError(f"invalid success: {arm} {episode_id}")
    spl = raw.get("spl")
    if isinstance(spl, bool) or not isinstance(spl, (int, float)) or not math.isfinite(spl) or not 0 <= spl <= 1:
        raise ValueError(f"invalid SPL: {arm} {episode_id}")
    return int(success), float(spl)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.result_root / "suite.completed").exists():
        raise ValueError("normalized terminal suite is incomplete")
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    if manifest["split"] != "val_seen" or len(ids) != 256 or len(set(ids)) != 256 or len(scenes) != 256 or len(set(scenes)) != 38:
        raise ValueError("unexpected manifest coverage")

    for arm in ARMS:
        if not (args.result_root / f"{arm}.completed").exists():
            raise ValueError(f"incomplete arm: {arm}")
        for shard in range(4):
            folder = args.result_root / arm / f"shard_{shard:02d}" / "log"
            expected = {f"stats_{ids[i]}_0.json" for i in range(shard, 256, 4)}
            actual = {p.name for p in folder.glob("stats_*_0.json")}
            if actual != expected:
                raise ValueError(f"shard coverage mismatch: {arm} shard {shard}; missing={len(expected-actual)} extra={len(actual-expected)}")

    success_count = {arm: 0 for arm in ARMS}
    spl_sum = {arm: 0.0 for arm in ARMS}
    paired_by_scene = defaultdict(list)
    discordance = [0, 0]
    rows = []
    for index, (episode_id, scene_id) in enumerate(zip(ids, scenes)):
        row = {"episode_id": episode_id, "scene_id": scene_id}
        values = {}
        for arm in ARMS:
            path = args.result_root / arm / f"shard_{index % 4:02d}" / "log" / f"stats_{episode_id}_0.json"
            raw = json.loads(path.read_text())
            values[arm] = metric(raw, episode_id, arm)
            success_count[arm] += values[arm][0]
            spl_sum[arm] += values[arm][1]
            row[arm] = {key: raw.get(key) for key in FIELDS}
        c_success, c_spl = values[CONTROL]
        t_success, t_spl = values[CANDIDATE]
        paired_by_scene[scene_id].append((t_success-c_success, t_spl-c_spl))
        discordance[0] += int(t_success == 1 and c_success == 0)
        discordance[1] += int(t_success == 0 and c_success == 1)
        rows.append(row)
    if discordance[0]-discordance[1] != success_count[CANDIDATE]-success_count[CONTROL]:
        raise ValueError("discordance mismatch")

    compact = {"schema": "normalized_terminal_val_seen256_compact_v1", "split": "val_seen", "manifest_sha256": MANIFEST_SHA256, "rows": rows}
    args.compact.write_text(json.dumps(compact, indent=2) + "\n")

    rng = random.Random(2026100604)
    scene_names = sorted(paired_by_scene)
    draws = []
    for _ in range(10000):
        selected = rng.choices(scene_names, k=len(scene_names))
        values = [item for scene in selected for item in paired_by_scene[scene]]
        draws.append((100*sum(x[0] for x in values)/len(values), 100*sum(x[1] for x in values)/len(values)))
    sr_delta = 100*(success_count[CANDIDATE]-success_count[CONTROL])/256
    spl_delta = 100*(spl_sum[CANDIDATE]-spl_sum[CONTROL])/256
    recount = {
        "schema": "normalized_terminal_val_seen256_independent_recount_v1",
        "split": "val_seen", "training_seed": 11, "episodes": 256, "scenes": 38,
        "manifest_sha256": MANIFEST_SHA256,
        "compact_sha256": hashlib.sha256(args.compact.read_bytes()).hexdigest(),
        "control": CONTROL, "candidate": CANDIDATE,
        "control_successes": success_count[CONTROL], "candidate_successes": success_count[CANDIDATE],
        "control_sr": success_count[CONTROL]/256, "candidate_sr": success_count[CANDIDATE]/256,
        "control_spl": spl_sum[CONTROL]/256, "candidate_spl": spl_sum[CANDIDATE]/256,
        "paired_sr_points": sr_delta, "paired_spl_points": spl_delta,
        "candidate_only_success": discordance[0], "control_only_success": discordance[1],
        "scene_cluster_bootstrap_95pct_sr_points": [sorted(x[0] for x in draws)[249], sorted(x[0] for x in draws)[9749]],
        "scene_cluster_bootstrap_95pct_spl_points": [sorted(x[1] for x in draws)[249], sorted(x[1] for x in draws)[9749]],
        "advance_gate_passed": sr_delta >= 2.0 and spl_delta >= 2.0,
        "inference_errors": 0,
        "interpretation": "One-seed val-seen development comparison; scene-cluster intervals are descriptive only.",
    }
    args.output.write_text(json.dumps(recount, indent=2) + "\n")
    print(json.dumps(recount, indent=2))


if __name__ == "__main__":
    main()
