"""Independently recount the frozen 778-episode GAE/control development screen."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


MANIFEST_SHA256 = "03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed"
CONTROL = "qwen3_exact_control_64step_seed11"
CANDIDATE = "turn_gae_64step_seed11"
ARMS = (CONTROL, CANDIDATE)
SHARDS = 4
FIELDS = ("episode_id", "success", "spl", "early_stop_reason", "distance_to_goal", "path_length", "oracle_success")


def metric(raw: dict, episode_id: str, arm: str) -> tuple[int, float]:
    declared = raw.get("episode_id")
    if declared is not None and str(declared) != episode_id:
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
        raise ValueError("GAE suite is incomplete")
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    if manifest["split"] != "val_seen" or len(ids) != 778 or len(set(ids)) != 778 or len(scenes) != 778 or len(set(scenes)) != 53:
        raise ValueError("unexpected manifest coverage")

    for arm in ARMS:
        if not (args.result_root / f"{arm}.completed").exists():
            raise ValueError(f"incomplete arm: {arm}")
        validation = json.loads((args.result_root / f"{arm}.validated.json").read_text())
        if validation["label"] != arm or validation["episodes"] != 778 or validation["inference_errors"] != 0:
            raise ValueError(f"invalid validator record: {arm}")
        for shard in range(SHARDS):
            folder = args.result_root / arm / f"shard_{shard:02d}" / "log"
            expected = {f"stats_{ids[i]}_0.json" for i in range(shard, 778, SHARDS)}
            actual = {path.name for path in folder.glob("stats_*_0.json")}
            if actual != expected:
                raise ValueError(f"shard coverage mismatch: {arm} shard {shard}; missing={len(expected-actual)} extra={len(actual-expected)}")

    successes = {arm: 0 for arm in ARMS}
    spl_sums = {arm: 0.0 for arm in ARMS}
    by_scene = defaultdict(list)
    discordance = [0, 0]
    rows = []
    for index, (episode_id, scene_id) in enumerate(zip(ids, scenes)):
        row = {"episode_id": episode_id, "scene_id": scene_id}
        values = {}
        for arm in ARMS:
            path = args.result_root / arm / f"shard_{index % SHARDS:02d}" / "log" / f"stats_{episode_id}_0.json"
            raw = json.loads(path.read_text())
            values[arm] = metric(raw, episode_id, arm)
            successes[arm] += values[arm][0]
            spl_sums[arm] += values[arm][1]
            row[arm] = {key: raw.get(key) for key in FIELDS}
        control_success, control_spl = values[CONTROL]
        candidate_success, candidate_spl = values[CANDIDATE]
        by_scene[scene_id].append((candidate_success - control_success, candidate_spl - control_spl))
        discordance[0] += int(candidate_success == 1 and control_success == 0)
        discordance[1] += int(candidate_success == 0 and control_success == 1)
        rows.append(row)
    if discordance[0] - discordance[1] != successes[CANDIDATE] - successes[CONTROL]:
        raise ValueError("discordance recount mismatch")
    for arm in ARMS:
        validation = json.loads((args.result_root / f"{arm}.validated.json").read_text())
        if validation["successes"] != successes[arm]:
            raise ValueError(f"validator success mismatch: {arm}")

    compact = {"schema": "multimodal_gae_val_seen778_compact_v1", "split": "val_seen", "manifest_sha256": MANIFEST_SHA256, "rows": rows}
    args.compact.write_text(json.dumps(compact, indent=2) + "\n")
    rng = random.Random(2026100605)
    scene_names = sorted(by_scene)
    draws = []
    for _ in range(10000):
        sampled = rng.choices(scene_names, k=len(scene_names))
        values = [item for scene in sampled for item in by_scene[scene]]
        draws.append((100 * sum(x[0] for x in values) / len(values), 100 * sum(x[1] for x in values) / len(values)))
    sr_delta = 100 * (successes[CANDIDATE] - successes[CONTROL]) / 778
    spl_delta = 100 * (spl_sums[CANDIDATE] - spl_sums[CONTROL]) / 778
    result = {
        "schema": "multimodal_gae_val_seen778_independent_recount_v1",
        "training_seed": 11, "episodes": 778, "scenes": 53,
        "manifest_sha256": MANIFEST_SHA256,
        "compact_sha256": hashlib.sha256(args.compact.read_bytes()).hexdigest(),
        "control": CONTROL, "candidate": CANDIDATE,
        "control_successes": successes[CONTROL], "candidate_successes": successes[CANDIDATE],
        "control_spl": spl_sums[CONTROL] / 778, "candidate_spl": spl_sums[CANDIDATE] / 778,
        "paired_sr_points": sr_delta, "paired_spl_points": spl_delta,
        "candidate_only_success": discordance[0], "control_only_success": discordance[1],
        "scene_cluster_bootstrap_95pct_sr_points": [sorted(x[0] for x in draws)[249], sorted(x[0] for x in draws)[9749]],
        "scene_cluster_bootstrap_95pct_spl_points": [sorted(x[1] for x in draws)[249], sorted(x[1] for x in draws)[9749]],
        "advance_gate_passed": sr_delta >= 2.0 and spl_delta >= 2.0,
        "inference_errors": 0,
        "interpretation": "Adaptively reused val-seen development split; one seed and decode, not unseen-scene evidence.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
