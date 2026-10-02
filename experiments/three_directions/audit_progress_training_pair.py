"""Check that a progress-reward run changes reward, not its matched train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


DATASETS = {
    64: ("branch_pilot_train.parquet", "a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3"),
    128: ("branch_scale512_train.parquet", "2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea"),
}
SEEDS = (11, 22, 33)


def read_rollouts(checkpoint_dir: Path, run_dir: Path, steps: int,
                  require_completed: bool = True) -> tuple[list[set[str]], dict]:
    if require_completed:
        assert (run_dir / "completed").exists()
    assert not (run_dir / "failed").exists()
    assert (run_dir / "validation.json").is_file()
    assert (checkpoint_dir / f"global_step_{steps}/actor/huggingface/config.json").is_file()
    path = checkpoint_dir / "rollout.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == steps
    assert [row["step"] for row in rows] == list(range(1, steps + 1))
    all_ids = set()
    per_step = []
    components_seen = 0
    nonzero_progress = 0
    invalid_distances = 0
    for row in rows:
        counts = Counter(str(item["episode_id"]) for item in row["info"])
        assert len(counts) == 4 and set(counts.values()) == {2}
        ids = set(counts)
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
        per_step.append(ids)
        for item in row["info"]:
            assert int(item["env_global_step"]) == int(item["env_local_step"])
            parts = item.get("reward_components", {})
            if "geodesic_progress_reward" in parts:
                bonus = float(parts["geodesic_progress_reward"])
                assert math.isfinite(bonus) and -1 <= bonus <= 1
                if item.get("progress_distance_invalid"):
                    assert bonus == 0
                components_seen += 1
                nonzero_progress += bonus != 0.0
                invalid_distances += bool(item.get("progress_distance_invalid"))
    return per_step, {
        "unique_episode_count": len(all_ids),
        "episode_ids": sorted(all_ids),
        "rollouts": len(rows) * 8,
        "progress_components_seen": components_seen,
        "nonzero_progress_rollouts": nonzero_progress,
        "invalid_progress_distances": invalid_distances,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--progress-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--steps", type=int, choices=DATASETS, default=128)
    parser.add_argument("--candidate-not-completed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_name, dataset_sha = DATASETS[args.steps]
    dataset = args.source_root / "data" / dataset_name
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert digest == dataset_sha
    table = pd.read_parquet(dataset)
    expected_ids = [str(item["episode_id"]) for item in table["extra_info"]]
    assert len(expected_ids) == len(set(expected_ids)) == args.steps * 4
    assert all(item["split"] == "train" for item in table["extra_info"])

    suffix = "" if args.seed == 11 else f"_seed{args.seed}"
    progress_name = f"three_directions_progress_fallback_{args.steps}step{suffix}"
    control_name = f"three_directions_branch_control_{args.steps}step{suffix}"
    progress_run = args.progress_root / "runlogs" / progress_name
    control_run = args.source_root / "runlogs" / control_name
    progress_config = (progress_run / "config.txt").read_text()
    control_config = (control_run / "config.txt").read_text()
    assert f"mode=progress_fallback steps={args.steps} " in progress_config
    assert f"mode=branch_control steps={args.steps} " in control_config
    assert f"seed={args.seed} " in progress_config and f"seed={args.seed} " in control_config
    assert f"dataset=data/{dataset_name} " in progress_config
    assert f"dataset=data/{dataset_name} " in control_config
    assert f"dataset_sha256={dataset_sha} " in progress_config

    progress_steps, progress = read_rollouts(
        args.progress_root / "verl_checkpoints" / progress_name, progress_run,
        args.steps, require_completed=not args.candidate_not_completed)
    control_steps, control = read_rollouts(
        args.source_root / "verl_checkpoints" / control_name, control_run,
        args.steps)
    assert progress_steps == control_steps
    assert progress["episode_ids"] == control["episode_ids"] == sorted(expected_ids)
    assert progress["progress_components_seen"] == progress["rollouts"]
    assert progress["nonzero_progress_rollouts"] > 0
    assert control["progress_components_seen"] == 0
    for value in (progress, control):
        del value["episode_ids"]
    result = {
        "seed": args.seed,
        "dataset_sha256": digest,
        "steps": args.steps,
        "matched_train_episode_sets_at_each_step": args.steps,
        "progress": progress,
        "destination_only_control": control,
        "interpretation": "Train-row and rollout-component audit, not held-out navigation evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
