"""Independently recount a conditional three-seed full val-unseen replication.

The frozen manifest and same-seed controls are read from raw shard stats.
One compact JSONL file per seed contains only episode metrics. The seed/scene
bootstrap is descriptive because this split was used during development.
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

MANIFEST_SHA256 = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SEEDS = (11, 22, 33)
FIELDS = ("success", "spl", "early_stop_reason", "distance_to_goal", "oracle_success")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_metric(raw: dict, episode_id: str, label: str) -> tuple[int, float]:
    if str(raw.get("episode_id", episode_id)) != episode_id:
        raise ValueError(f"episode mismatch: {label} {episode_id}")
    if raw.get("early_stop_reason") == "inference_error":
        raise ValueError(f"inference error: {label} {episode_id}")
    success = raw.get("success")
    if isinstance(success, bool):
        success = int(success)
    if success not in (0, 0.0, 1, 1.0):
        raise ValueError(f"invalid success: {label} {episode_id}")
    spl = raw.get("spl")
    if isinstance(spl, bool) or not isinstance(spl, (int, float)) or not math.isfinite(spl) or not 0 <= spl <= 1:
        raise ValueError(f"invalid SPL: {label} {episode_id}")
    return int(success), float(spl)


def check_arm(root: Path, label: str, ids: list[str]) -> dict:
    if not (root / f"{label}.completed").exists():
        raise ValueError(f"arm incomplete: {label}")
    if (root / f"{label}.failed").exists():
        raise ValueError(f"failed marker: {label}")
    validator_path = root / f"{label}.validated.json"
    validator = json.loads(validator_path.read_text())
    if validator.get("label") != label or validator.get("episodes") != len(ids) or validator.get("inference_errors") != 0:
        raise ValueError(f"invalid final validator: {label}")
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}" / "log"
        expected = {f"stats_{ids[i]}_0.json" for i in range(shard, len(ids), 4)}
        observed = {p.name for p in folder.glob("stats_*_0.json")}
        if observed != expected:
            raise ValueError(f"shard coverage: {label} shard {shard}; missing={len(expected-observed)} extra={len(observed-expected)}")
    return {"label": label, "validator_sha256": sha(validator_path), "validated_successes": validator["successes"]}


def read_seed(
    seed: int, ids: list[str], scenes: list[str], candidate_root: Path,
    candidate_label: str, control_root: Path, control_label: str,
    compact_dir: Path,
) -> tuple[dict, dict[str, list[tuple[int, float]]]]:
    candidate_validation = check_arm(candidate_root, candidate_label, ids)
    control_validation = check_arm(control_root, control_label, ids)
    roots = {"candidate": (candidate_root, candidate_label), "control": (control_root, control_label)}
    successes = {name: 0 for name in roots}
    spl_sums = {name: 0.0 for name in roots}
    candidate_only = control_only = 0
    by_scene = defaultdict(list)
    rows = []
    for index, (episode_id, scene_id) in enumerate(zip(ids, scenes)):
        row = {"episode_id": episode_id, "scene_id": scene_id}
        values = {}
        for name, (root, label) in roots.items():
            path = root / label / f"shard_{index % 4:02d}" / "log" / f"stats_{episode_id}_0.json"
            raw = json.loads(path.read_text())
            values[name] = checked_metric(raw, episode_id, label)
            successes[name] += values[name][0]
            spl_sums[name] += values[name][1]
            row[name] = {key: raw.get(key) for key in FIELDS}
        ds = values["candidate"][0] - values["control"][0]
        dp = values["candidate"][1] - values["control"][1]
        by_scene[scene_id].append((ds, dp))
        candidate_only += int(ds == 1)
        control_only += int(ds == -1)
        rows.append(row)
    if successes["candidate"] != candidate_validation["validated_successes"] or successes["control"] != control_validation["validated_successes"]:
        raise ValueError(f"raw and validated successes disagree: seed {seed}")
    if candidate_only-control_only != successes["candidate"]-successes["control"]:
        raise ValueError(f"discordance recount mismatch: seed {seed}")
    compact_path = compact_dir / f"seed{seed}_paired_episodes.jsonl"
    compact_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    n = len(ids)
    result = {
        "seed": seed, "episodes": n, "scenes": len(set(scenes)),
        "candidate": candidate_validation, "control": control_validation,
        "candidate_successes": successes["candidate"], "control_successes": successes["control"],
        "candidate_sr": successes["candidate"] / n, "control_sr": successes["control"] / n,
        "candidate_spl": spl_sums["candidate"] / n, "control_spl": spl_sums["control"] / n,
        "paired_sr_points": 100*(successes["candidate"]-successes["control"])/n,
        "paired_spl_points": 100*(spl_sums["candidate"]-spl_sums["control"])/n,
        "candidate_only_success": candidate_only, "control_only_success": control_only,
        "compact_sha256": sha(compact_path),
    }
    return result, by_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--candidate-pattern", required=True, help="label with {seed} placeholder")
    ap.add_argument("--control-root", type=Path, required=True)
    ap.add_argument("--control-pattern", default="oracle_exact512_control_128_seed{seed}")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--compact-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if "{seed}" not in args.candidate_pattern or "{seed}" not in args.control_pattern:
        raise ValueError("both label patterns must contain {seed}")
    if sha(args.manifest) != MANIFEST_SHA256:
        raise ValueError("full manifest SHA-256 mismatch")
    manifest = json.loads(args.manifest.read_text())
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    if manifest["split"] != "val_unseen" or len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != len(ids) or len(set(scenes)) != 11:
        raise ValueError("unexpected val-unseen manifest coverage")
    args.compact_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    scenes_by_seed = {}
    for seed in SEEDS:
        result, by_scene = read_seed(seed, ids, scenes, args.candidate_root,
                                     args.candidate_pattern.format(seed=seed),
                                     args.control_root, args.control_pattern.format(seed=seed),
                                     args.compact_dir)
        pairs.append(result)
        scenes_by_seed[seed] = by_scene
    rng = random.Random(20261006128)
    draws = []
    for _ in range(10000):
        selected_seeds = rng.choices(SEEDS, k=len(SEEDS))
        seed_means = []
        for seed in selected_seeds:
            scene_names = sorted(scenes_by_seed[seed])
            selected_scenes = rng.choices(scene_names, k=len(scene_names))
            values = [item for scene in selected_scenes for item in scenes_by_seed[seed][scene]]
            seed_means.append((100*sum(x[0] for x in values)/len(values),
                               100*sum(x[1] for x in values)/len(values)))
        draws.append((statistics.mean(x[0] for x in seed_means),
                      statistics.mean(x[1] for x in seed_means)))
    sr_deltas = [p["paired_sr_points"] for p in pairs]
    spl_deltas = [p["paired_spl_points"] for p in pairs]
    sr_draws = sorted(x[0] for x in draws)
    spl_draws = sorted(x[1] for x in draws)
    report = {
        "schema": "turn_credit_reward_three_seed_full1839_independent_recount_v1",
        "manifest_sha256": MANIFEST_SHA256, "seeds": list(SEEDS),
        "episodes_per_seed": 1839, "scenes": 11, "pairs": pairs,
        "mean_paired_sr_points": statistics.mean(sr_deltas),
        "sample_sd_paired_sr_points": statistics.stdev(sr_deltas),
        "mean_paired_spl_points": statistics.mean(spl_deltas),
        "sample_sd_paired_spl_points": statistics.stdev(spl_deltas),
        "seed_scene_bootstrap_95pct_sr_points": [sr_draws[249], sr_draws[9749]],
        "seed_scene_bootstrap_95pct_spl_points": [spl_draws[249], spl_draws[9749]],
        "inference_errors": 0,
        "interpretation": "Three-seed larger development replication; val-unseen was used in prior method development and the intervals are descriptive only.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mean_paired_sr_points": report["mean_paired_sr_points"],
                      "mean_paired_spl_points": report["mean_paired_spl_points"]}, indent=2))


if __name__ == "__main__":
    main()
