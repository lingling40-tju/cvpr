"""Independently recount one frozen R2R-train scene-screen comparison.

This script reads raw per-episode Habitat stats. It does not run inference,
choose episodes, or treat the train-scene screen as a clean final test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


SCREEN_HASHES = {
    "development": "8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3",
    "reserved": "412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241",
}
EPISODES = 256
SHARDS = 4


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm(root: Path, label: str, ids: list[str], *,
             require_completed: bool = True) -> dict[str, tuple[int, float]]:
    if (require_completed and not (root / f"{label}.completed").exists()) or \
            (root / f"{label}.failed").exists():
        raise ValueError(f"arm is incomplete or failed: {label}")
    rows = {}
    for shard in range(SHARDS):
        folder = root / label / f"shard_{shard:02d}" / "log"
        expected = {f"stats_{ids[index]}_0.json"
                    for index in range(shard, len(ids), SHARDS)}
        observed = {path.name for path in folder.glob("stats_*_0.json")}
        if observed != expected:
            raise ValueError(f"shard coverage differs: {label} shard {shard}")
        for index in range(shard, len(ids), SHARDS):
            episode_id = ids[index]
            raw = json.loads((folder / f"stats_{episode_id}_0.json").read_text())
            if str(raw.get("episode_id", episode_id)) != episode_id or \
                    raw.get("early_stop_reason") == "inference_error":
                raise ValueError(f"episode identity or inference error: {label} {episode_id}")
            success = raw.get("success")
            spl = raw.get("spl")
            if isinstance(success, bool):
                success = int(success)
            if success not in (0, 0.0, 1, 1.0) or \
                    isinstance(spl, bool) or not isinstance(spl, (int, float)) or \
                    not math.isfinite(spl) or not 0 <= spl <= 1:
                raise ValueError(f"invalid navigation metric: {label} {episode_id}")
            rows[episode_id] = (int(success), float(spl))
    if set(rows) != set(ids):
        raise ValueError(f"arm episode coverage differs: {label}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(SCREEN_HASHES), required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if digest(args.manifest) != SCREEN_HASHES[args.role]:
        raise ValueError("frozen train-scene manifest SHA-256 mismatch")
    manifest = json.loads(args.manifest.read_text())
    ids = [str(item) for item in manifest["episode_ids"]]
    scenes = [str(item) for item in manifest["scene_ids"]]
    if manifest["schema"] != "trajectory_sil_rl_train_screen_v1" or \
            manifest["split"] != "train" or manifest["role"] != args.role or \
            len(ids) != EPISODES or len(set(ids)) != EPISODES or \
            len(scenes) != EPISODES or len(set(scenes)) != 8:
        raise ValueError("unexpected train-scene manifest coverage")
    if args.control == args.candidate:
        raise ValueError("control and candidate labels must differ")
    control = load_arm(args.root, args.control, ids)
    candidate = load_arm(args.root, args.candidate, ids)

    paired = []
    by_scene = defaultdict(list)
    for episode_id, scene_id in zip(ids, scenes):
        cs, cp = control[episode_id]
        ts, tp = candidate[episode_id]
        row = {"episode_id": episode_id, "scene_id": scene_id,
               "control_success": cs, "candidate_success": ts,
               "control_spl": cp, "candidate_spl": tp}
        paired.append(row)
        by_scene[scene_id].append((ts - cs, tp - cp))

    rng = random.Random(20261006128)
    scene_names = sorted(by_scene)
    draws = []
    for _ in range(10000):
        sampled = [pair for scene in rng.choices(scene_names, k=len(scene_names))
                   for pair in by_scene[scene]]
        draws.append((100 * statistics.mean(value[0] for value in sampled),
                      100 * statistics.mean(value[1] for value in sampled)))
    sr_draws = sorted(value[0] for value in draws)
    spl_draws = sorted(value[1] for value in draws)
    control_successes = sum(row["control_success"] for row in paired)
    candidate_successes = sum(row["candidate_success"] for row in paired)
    control_spl = statistics.mean(row["control_spl"] for row in paired)
    candidate_spl = statistics.mean(row["candidate_spl"] for row in paired)
    candidate_only = sum(row["candidate_success"] == 1 and
                         row["control_success"] == 0 for row in paired)
    control_only = sum(row["control_success"] == 1 and
                       row["candidate_success"] == 0 for row in paired)
    if candidate_only - control_only != candidate_successes - control_successes:
        raise ValueError("paired success discordance does not reconcile")
    args.compact.parent.mkdir(parents=True, exist_ok=True)
    args.compact.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in paired))
    report = {
        "schema": "positive_trajectory_train_scene_pair_v1",
        "role": args.role, "split": "train", "manifest_sha256": SCREEN_HASHES[args.role],
        "episodes": EPISODES, "scenes": len(scene_names),
        "control": args.control, "candidate": args.candidate,
        "control_successes": control_successes,
        "candidate_successes": candidate_successes,
        "control_sr": control_successes / EPISODES,
        "candidate_sr": candidate_successes / EPISODES,
        "control_spl": control_spl,
        "candidate_spl": candidate_spl,
        "paired_sr_points": 100 * (candidate_successes - control_successes) / EPISODES,
        "paired_spl_points": 100 * (candidate_spl - control_spl),
        "candidate_only_success": candidate_only,
        "control_only_success": control_only,
        "scene_bootstrap_95pct_sr_points": [sr_draws[249], sr_draws[9749]],
        "scene_bootstrap_95pct_spl_points": [spl_draws[249], spl_draws[9749]],
        "compact_sha256": digest(args.compact),
        "inference_errors": 0,
        "interpretation": "One-seed RL-stage scene-disjoint development comparison; prior SFT and exploratory work may have seen these train scenes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"paired_sr_points": report["paired_sr_points"],
                      "paired_spl_points": report["paired_spl_points"]}))


if __name__ == "__main__":
    main()
