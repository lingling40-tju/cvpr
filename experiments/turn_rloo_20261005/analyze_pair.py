"""Recount paired val-seen results with exact episode and scene coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def load_label(root: Path, label: str, ids: list[str], shards: int) -> dict[str, dict]:
    rows = {}
    for shard in range(shards):
        folder = root / label / f"shard_{shard:02d}" / "log"
        for episode_id in ids[shard::shards]:
            path = folder / f"stats_{episode_id}_0.json"
            row = json.loads(path.read_text())
            if str(row.get("episode_id", episode_id)) != episode_id:
                raise ValueError(f"episode ID mismatch in {path}")
            if row.get("early_stop_reason") == "inference_error":
                raise ValueError(f"inference error in {path}")
            if episode_id in rows:
                raise ValueError(f"duplicate episode {episode_id}")
            rows[episode_id] = row
    if set(rows) != set(ids):
        raise ValueError("incomplete model coverage")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("control")
    ap.add_argument("candidate")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    if manifest["split"] != "val_seen" or len(manifest["episode_ids"]) != 256:
        raise ValueError("unexpected fixed val-seen manifest")
    ids = [str(x) for x in manifest["episode_ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate manifest episode")
    c = load_label(args.root, args.control, ids, args.shards)
    t = load_label(args.root, args.candidate, ids, args.shards)
    scenes = {str(manifest["scene_ids"][i]): [] for i in range(len(ids))}
    paired = []
    for episode_id, scene in zip(ids, manifest["scene_ids"]):
        row = {
            "episode_id": episode_id,
            "scene_id": str(scene),
            "control_success": bool(c[episode_id]["success"]),
            "candidate_success": bool(t[episode_id]["success"]),
            "control_spl": float(c[episode_id]["spl"]),
            "candidate_spl": float(t[episode_id]["spl"]),
        }
        scenes[str(scene)].append(row)
        paired.append(row)
    control_sr = sum(x["control_success"] for x in paired) / len(paired)
    candidate_sr = sum(x["candidate_success"] for x in paired) / len(paired)
    control_spl = statistics.mean(x["control_spl"] for x in paired)
    candidate_spl = statistics.mean(x["candidate_spl"] for x in paired)
    # Descriptive cluster bootstrap only; development data and one seed.
    rng = random.Random(20261005)
    scene_ids = sorted(scenes)
    samples = []
    for _ in range(2000):
        sample = [r for scene in rng.choices(scene_ids, k=len(scene_ids)) for r in scenes[scene]]
        samples.append((statistics.mean(x["candidate_success"] - x["control_success"] for x in sample),
                        statistics.mean(x["candidate_spl"] - x["control_spl"] for x in sample)))
    def interval(col: int) -> list[float]:
        values = sorted(x[col] for x in samples)
        return [100 * values[49], 100 * values[1949]]
    result = {
        "schema": "turn_rloo_val_seen256_paired_v1",
        "split": "val_seen",
        "episodes": len(ids),
        "scenes": len(scene_ids),
        "control": args.control,
        "candidate": args.candidate,
        "control_successes": sum(x["control_success"] for x in paired),
        "candidate_successes": sum(x["candidate_success"] for x in paired),
        "control_sr": control_sr,
        "candidate_sr": candidate_sr,
        "paired_sr_points": 100 * (candidate_sr - control_sr),
        "control_spl": control_spl,
        "candidate_spl": candidate_spl,
        "paired_spl_points": 100 * (candidate_spl - control_spl),
        "candidate_only_success": sum(x["candidate_success"] and not x["control_success"] for x in paired),
        "control_only_success": sum(x["control_success"] and not x["candidate_success"] for x in paired),
        "scene_bootstrap_95pct_sr_points": interval(0),
        "scene_bootstrap_95pct_spl_points": interval(1),
        "interpretation": "One-seed val-seen development comparison; not unseen-scene generalization or a deployable semantic-reward result.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
