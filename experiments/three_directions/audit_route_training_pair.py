"""Audit terminal route-fidelity reward and matched training rows."""

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
    nonzero_ndtw = 0
    budget_nonzero_ndtw = 0
    malformed_rollouts = 0
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
            if "ndtw_reward" in parts:
                bonus = float(parts["ndtw_reward"])
                assert math.isfinite(bonus) and 0 <= bonus <= 1
                reason = item["end_reason"]
                if reason == "unexpected format.":
                    assert bonus == 0
                    malformed_rollouts += 1
                components_seen += 1
                nonzero_ndtw += bonus != 0.0
                if reason in ("number of turns exceeded.", "number of steps exceeded."):
                    budget_nonzero_ndtw += bonus != 0.0
    return per_step, {
        "unique_episode_count": len(all_ids),
        "episode_ids": sorted(all_ids),
        "rollouts": len(rows) * 8,
        "ndtw_components_seen": components_seen,
        "nonzero_ndtw_rollouts": nonzero_ndtw,
        "budget_exhausted_nonzero_ndtw_rollouts": budget_nonzero_ndtw,
        "malformed_rollouts": malformed_rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
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
    route_name = f"three_directions_route_fidelity_{args.steps}step{suffix}"
    control_name = f"three_directions_branch_control_{args.steps}step{suffix}"
    route_run = args.route_root / "runlogs" / route_name
    control_run = args.source_root / "runlogs" / control_name
    route_config = (route_run / "config.txt").read_text()
    control_config = (control_run / "config.txt").read_text()
    assert f"mode=route_fidelity steps={args.steps} " in route_config
    assert f"mode=branch_control steps={args.steps} " in control_config
    assert f"seed={args.seed} " in route_config and f"seed={args.seed} " in control_config
    assert f"dataset=data/{dataset_name} " in route_config
    assert f"dataset=data/{dataset_name} " in control_config
    assert f"dataset_sha256={dataset_sha} " in route_config

    route_steps, route = read_rollouts(
        args.route_root / "verl_checkpoints" / route_name, route_run,
        args.steps, require_completed=not args.candidate_not_completed)
    control_steps, control = read_rollouts(
        args.source_root / "verl_checkpoints" / control_name, control_run,
        args.steps)
    assert route_steps == control_steps
    assert route["episode_ids"] == control["episode_ids"] == sorted(expected_ids)
    assert route["ndtw_components_seen"] == route["rollouts"]
    assert route["nonzero_ndtw_rollouts"] > 0
    assert route["budget_exhausted_nonzero_ndtw_rollouts"] > 0
    assert control["ndtw_components_seen"] == control["rollouts"]
    assert control["nonzero_ndtw_rollouts"] == 0
    for value in (route, control):
        del value["episode_ids"]
    service_log = args.route_root / "runlogs" / "route_service.log"
    nonfinite_warnings = sum(
        "nonfinite generated nDTW" in line for line in service_log.open(errors="replace")
    )
    result = {
        "seed": args.seed,
        "dataset_sha256": digest,
        "steps": args.steps,
        "matched_train_episode_sets_at_each_step": args.steps,
        "route_fidelity": route,
        "destination_only_control": control,
        "nonfinite_ndtw_service_warnings": nonfinite_warnings,
        "interpretation": "Train-row and rollout-component audit, not held-out navigation evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
