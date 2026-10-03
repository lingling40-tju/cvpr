"""Describe terminal behavior for the matched failure-only online screen.

This analysis is post-hoc. It tests an incentive hypothesis but does not
establish that any particular behavior caused the navigation regression.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path


def load_label(root: Path, label: str, ids: list[str]) -> dict[str, dict]:
    if not (root / f"{label}.completed").is_file():
        raise ValueError(f"incomplete label {label}")
    rows = {}
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        expected = ids[shard::4]
        observed = list(map(str, summary["episode_ids"]))
        if summary["count"] != len(expected) or set(observed) != set(expected) or \
                len(observed) != len(set(observed)) or summary["inference_errors"]:
            raise ValueError(f"invalid shard {label}/{shard}")
        for eid in expected:
            row = json.loads((folder / "log" / f"stats_{eid}_0.json").read_text())
            if str(row["id"]) != eid or row.get("early_stop_reason") == "inference_error":
                raise ValueError(f"invalid episode {label}/{eid}")
            rows[eid] = row
    if set(rows) != set(ids):
        raise ValueError(f"episode coverage mismatch {label}")
    return rows


def summary(rows: dict[str, dict], ids: list[str]) -> dict:
    return {"episodes": len(ids),
            "successes": sum(bool(rows[i]["success"]) for i in ids),
            "oracle_successes": sum(bool(rows[i]["oracle_success"]) for i in ids),
            "early_stop_reasons": dict(Counter(str(rows[i].get("early_stop_reason")) for i in ids)),
            "mean_path_length_m": statistics.mean(float(rows[i]["path_length"]) for i in ids),
            "mean_distance_to_goal_m": statistics.mean(float(rows[i]["distance_to_goal"]) for i in ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--candidate", default="failure_only_group4_64_seed11")
    parser.add_argument("--control", default="group4_64_seed11")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.eval_root / "manifest.json").read_text())
    ids = list(map(str, manifest["episode_ids"]))
    if len(ids) != len(set(ids)) or len(ids) != 256 or len(set(manifest["scene_ids"])) != 11:
        raise ValueError("unexpected fixed evaluation manifest")
    candidate = load_label(args.eval_root, args.candidate, ids)
    control = load_label(args.eval_root, args.control, ids)
    discordant = {
        "candidate_only": [i for i in ids if candidate[i]["success"] and not control[i]["success"]],
        "control_only": [i for i in ids if control[i]["success"] and not candidate[i]["success"]],
        "both_failure": [i for i in ids if not candidate[i]["success"] and not control[i]["success"]],
    }
    groups = {}
    for name, selected in discordant.items():
        groups[name] = {"episodes": len(selected),
                        "candidate_reasons": dict(Counter(str(candidate[i].get("early_stop_reason")) for i in selected)),
                        "control_reasons": dict(Counter(str(control[i].get("early_stop_reason")) for i in selected))}
    train = [json.loads(line) for line in args.train_rollout.read_text().splitlines()]
    if len(train) != 64 or [row["step"] for row in train] != list(range(1, 65)):
        raise ValueError("incomplete training rollout")
    trajectories = [item for row in train for item in row["info"]]
    if len(trajectories) != 1024:
        raise ValueError("unexpected training rollout count")
    failure = [item for item in trajectories if not item["task_success"]]
    terminal = {}
    for reason in sorted({item["end_reason"] for item in failure}):
        selected = [item for item in failure if item["end_reason"] == reason]
        terminal[reason] = {"rollouts": len(selected),
                            "mean_failure_bonus": statistics.mean(
                                float(item["reward_components"]["fused_bonus"])
                                for item in selected),
                            "terminal_distance_below_3_5_m": sum(
                                float(item["distance_to_goal"]) < 3.5
                                for item in selected)}
    result = {"schema": "failure_only_matched_val256_modes_v1",
              "interpretation": "Post-hoc behavior description on the fixed 256 val-unseen episodes and the candidate's training rollouts. It does not identify a causal mechanism or prove a new reward will improve navigation.",
              "candidate": args.candidate, "control": args.control,
              "candidate_val": summary(candidate, ids),
              "control_val": summary(control, ids),
              "paired_outcome_groups": groups,
              "candidate_train_failure_terminal_modes": terminal}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
