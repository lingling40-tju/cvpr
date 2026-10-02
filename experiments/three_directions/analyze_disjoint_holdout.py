"""Analyze the 1,583 val-unseen episodes unused by the fixed 256 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from analyze_direction_eval import scene_bootstrap, summarize
from analyze_scaled_val256 import load_label, scene_seed_bootstrap


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46"
SEEDS = (11, 22, 33)


def read_manifest(path: Path, expected_sha: str) -> tuple[list[str], list[str]]:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    data = json.loads(path.read_text())
    ids = [str(item) for item in data["episode_ids"]]
    scenes = [str(item) for item in data["scene_ids"]]
    assert len(ids) == len(set(ids)) == len(scenes)
    return ids, scenes


def analyze(root: Path, screen_manifest: Path, mode: str) -> dict:
    full_ids, full_scenes = read_manifest(root / "manifest.json", FULL_SHA)
    screen_ids, screen_scenes = read_manifest(screen_manifest, SCREEN_SHA)
    assert len(full_ids) == 1839 and len(screen_ids) == 256
    full_scene_by_id = dict(zip(full_ids, full_scenes))
    assert set(screen_ids) < set(full_ids)
    assert all(full_scene_by_id[eid] == scene
               for eid, scene in zip(screen_ids, screen_scenes))
    screened = set(screen_ids)
    ids = [eid for eid in full_ids if eid not in screened]
    scenes = [full_scene_by_id[eid] for eid in ids]
    # The fixed screen includes all 18 episodes of one small scene, so the
    # disjoint remainder spans ten of the full benchmark's eleven scenes.
    assert len(ids) == 1583 and len(set(scenes)) == 10

    result = {
        "split": "val_unseen_disjoint_from_fixed256_screen",
        "episodes": 1583,
        "excluded_screen_episodes": 256,
        "scenes": 10,
        "full_manifest_sha256": FULL_SHA,
        "screen_manifest_sha256": SCREEN_SHA,
        "mode": mode,
        "train_steps": 128,
        "train_rows_per_arm": 512,
        "rollouts_per_episode": 2,
        "models": {},
        "paired_seed_differences": {},
        "interpretation": (
            "Exploratory fixed-screen-disjoint analysis; the 1,583 episodes "
            "were not used for the earlier 256-episode model screen and "
            "span ten scenes because the screen exhausted one small scene. "
            "Training seeds and stochastic decodes remain sources of uncertainty."
        ),
    }
    paired_rows = {}
    for seed in SEEDS:
        candidate_label = f"{mode}128_seed{seed}"
        control_label = f"branch_control128_seed{seed}"
        candidate = load_label(root, candidate_label, full_ids)
        control = load_label(root, control_label, full_ids)
        paired_rows[seed] = (candidate, control)
        result["models"][candidate_label] = summarize(candidate, ids)
        result["models"][control_label] = summarize(control, ids)
        result["paired_seed_differences"][str(seed)] = {
            "candidate_label": candidate_label,
            "control_label": control_label,
            "sr_pp": 100 * sum(bool(candidate[i]["success"]) - bool(control[i]["success"])
                               for i in ids) / len(ids),
            "spl_pp": 100 * sum(float(candidate[i]["spl"]) - float(control[i]["spl"])
                                for i in ids) / len(ids),
            "candidate_only_successes": sum(bool(candidate[i]["success"]) and
                                            not bool(control[i]["success"]) for i in ids),
            "control_only_successes": sum(bool(control[i]["success"]) and
                                          not bool(candidate[i]["success"]) for i in ids),
            "scene_cluster_bootstrap95": scene_bootstrap(candidate, control, ids, scenes),
        }
    for metric in ("sr_pp", "spl_pp"):
        values = [result["paired_seed_differences"][str(seed)][metric] for seed in SEEDS]
        result[f"mean_paired_{metric}"] = statistics.mean(values)
        result[f"sd_paired_{metric}"] = statistics.stdev(values)
    result["scene_seed_bootstrap95"] = scene_seed_bootstrap(paired_rows, ids, scenes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("branch", "progress"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.screen_manifest, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
