"""Audit three matched progress-reward seeds on one val-unseen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from analyze_direction_eval import scene_bootstrap, summarize
from analyze_scaled_val256 import load_label, scene_seed_bootstrap


MANIFEST_SHA = {
    256: "546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46",
    1839: "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e",
}
SEEDS = (11, 22, 33)


def analyze(root: Path, expected_count: int) -> dict:
    manifest_path = root / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == MANIFEST_SHA[expected_count]
    manifest = json.loads(manifest_path.read_text())
    ids = [str(item) for item in manifest["episode_ids"]]
    scenes = [str(item) for item in manifest["scene_ids"]]
    assert len(ids) == len(set(ids)) == len(scenes) == expected_count
    assert len(set(scenes)) == 11

    output = {
        "split": "val_unseen",
        "episodes": expected_count,
        "scenes": 11,
        "manifest_sha256": digest,
        "mode": "progress",
        "train_steps": 128,
        "train_rows_per_arm": 512,
        "rollouts_per_episode": 2,
        "models": {},
        "paired_seed_differences": {},
    }
    paired_rows = {}
    for seed in SEEDS:
        candidate_label = f"progress128_seed{seed}"
        control_label = f"branch_control128_seed{seed}"
        candidate = load_label(root, candidate_label, ids)
        control = load_label(root, control_label, ids)
        paired_rows[seed] = (candidate, control)
        output["models"][candidate_label] = summarize(candidate, ids)
        output["models"][control_label] = summarize(control, ids)
        output["paired_seed_differences"][str(seed)] = {
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
        values = [output["paired_seed_differences"][str(seed)][metric] for seed in SEEDS]
        output[f"mean_paired_{metric}"] = statistics.mean(values)
        output[f"sd_paired_{metric}"] = statistics.stdev(values)
    output["scene_seed_bootstrap95"] = scene_seed_bootstrap(paired_rows, ids, scenes)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, choices=MANIFEST_SHA, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
