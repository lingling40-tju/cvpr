"""Audit paired training episodes and intervention isolation for one seed."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def records(path: Path, required_steps: int, exact: bool) -> list[dict]:
    rows = []
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if exact:
                raise
            break
    if exact:
        assert len(rows) == required_steps
    else:
        assert len(rows) >= required_steps
    rows = rows[:required_steps]
    assert [row["step"] for row in rows] == list(range(1, required_steps + 1))
    return rows


def groups(rows: list[dict]) -> tuple[list[set[str]], set[str]]:
    per_step = []
    all_ids = set()
    for row in rows:
        counts = Counter(str(item["episode_id"]) for item in row["info"])
        assert len(counts) == 4 and set(counts.values()) == {2}, row["step"]
        ids = set(counts)
        assert not all_ids.intersection(ids), row["step"]
        all_ids.update(ids)
        per_step.append(ids)
    return per_step, all_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.seed in (11, 22, 33) and 1 <= args.steps <= 128

    data = pd.read_parquet(args.dataset)
    dataset_ids = [str(row["episode_id"]) for row in data["extra_info"]]
    assert len(dataset_ids) == len(set(dataset_ids)) == 512
    assert all(row["split"] == "train" for row in data["extra_info"])
    dataset_prefix = {
        str(row["episode_id"]): len(row["forced_history_actions"])
        for row in data["extra_info"]
    }
    result = {"seed": args.seed, "audited_steps": args.steps,
              "partial": args.partial, "dataset_rows": len(dataset_ids),
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "interaction_unit": "grouped action commands, including stop; a forward command can execute multiple simulator steps",
              "arms": {}}
    observed = {}
    for arm in ("branch", "branch_control"):
        name = f"three_directions_{arm}_128step"
        if args.seed != 11:
            name += f"_seed{args.seed}"
        run = args.root / "runlogs" / name
        checkout = args.root / "verl_checkpoints" / name
        if not args.partial:
            assert (run / "completed").exists() and not (run / "failed").exists()
            assert (run / "validation.json").exists()
            assert (checkout / "global_step_128/actor/huggingface/config.json").exists()
        config = (run / "config.txt").read_text().strip()
        assert f"mode={arm} " in config and f"seed={args.seed} " in config
        assert f"dataset={args.dataset}" in config
        rows = records(checkout / "rollout.jsonl", args.steps, not args.partial)
        per_step, all_ids = groups(rows)
        expected = set(dataset_ids[:4 * args.steps])
        assert all_ids == expected
        log = (run / "train.log").read_text(errors="replace")
        prefix_lengths = [int(x) for x in re.findall(r"num_gt_actions=(\d+)", log)]
        assert len(prefix_lengths) >= 4 * args.steps
        if arm == "branch":
            assert all(4 <= x <= 9 for x in prefix_lengths)
            assert "[ALTERNATIVE_PREFIX]" in log
        else:
            assert set(prefix_lengths) == {0}
            assert "[ALTERNATIVE_PREFIX]" not in log
        generated_commands = 0
        replayed_commands = 0
        for row in rows:
            for item in row["info"]:
                local = int(item["env_local_step"])
                global_count = int(item["env_global_step"])
                replayed = global_count - local
                assert local >= 0 and replayed >= 0
                expected_replay = dataset_prefix[str(item["episode_id"])] if arm == "branch" else 0
                assert replayed == expected_replay, (arm, row["step"], item["episode_id"])
                generated_commands += local
                replayed_commands += replayed
        observed[arm] = per_step
        result["arms"][arm] = {
            "episode_count": len(all_ids),
            "rollouts_per_episode": 2,
            "training_rollouts": len(rows) * 8,
            "prefix_action_count_min": min(prefix_lengths),
            "prefix_action_count_max": max(prefix_lengths),
            "generated_grouped_action_commands": generated_commands,
            "replayed_grouped_action_commands": replayed_commands,
            "total_grouped_action_commands": generated_commands + replayed_commands,
        }
        if not args.partial:
            started = dt.datetime.fromtimestamp((run / "config.txt").stat().st_mtime, dt.timezone.utc)
            finished = dt.datetime.fromisoformat((run / "completed").read_text().strip().replace("Z", "+00:00"))
            elapsed_hours = (finished - started).total_seconds() / 3600
            assert elapsed_hours > 0
            result["arms"][arm]["wrapper_elapsed_hours"] = elapsed_hours
            result["arms"][arm]["wrapper_elapsed_definition"] = (
                "config.txt mtime to completed marker; includes training, checkpoint save, and validation"
            )
    assert all(a == b for a, b in zip(observed["branch"], observed["branch_control"]))
    result["matched_episode_sets_at_each_step"] = args.steps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
