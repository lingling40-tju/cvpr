#!/usr/bin/env python3
"""Export executed-action diversity from audited, completed runs' first batches."""

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sampling-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    provenance = json.loads(args.sampling_provenance.read_text())
    if provenance["schema"] != "positive_sampling_seed_provenance_v1":
        raise ValueError("Sampling-provenance schema differs")
    for name, expected in provenance["source_sha256"].items():
        if digest(args.root / name) != expected:
            raise ValueError("Inspected source changed: {}".format(name))
    if not provenance["checked_runs"]:
        raise ValueError("No completed runs in provenance")
    runs, signatures = [], {}
    for run in provenance["checked_runs"]:
        label = run["run"]
        run_dir = args.root / "runlogs" / label
        if not (run_dir / "completed").is_file() or (run_dir / "failed").exists():
            raise ValueError("Run is not successfully completed")
        path = args.root / "verl_checkpoints" / label / "rollout.jsonl"
        if digest(path) != run["rollout_sha256"]:
            raise ValueError("Original rollout file changed")
        with path.open() as stream:
            first = json.loads(next(stream))
        if first["step"] != 1 or len(first["info"]) != 32:
            raise ValueError("First batch differs from frozen group-four batch")
        rows, groups = [], defaultdict(list)
        for item in first["info"]:
            by_turn = [turn["executed_actions"] for turn in item["gen_traj"]]
            if not by_turn or any(not isinstance(actions, list) for actions in by_turn):
                raise ValueError("Executed-action lists are unavailable")
            flat = [action for actions in by_turn for action in actions]
            row = {"episode_id": str(item["episode_id"]), "actions_by_turn": by_turn,
                   "flattened_actions": flat, "total_reward": item["total_reward"],
                   "task_success": item["task_success"], "distance_to_goal": item["distance_to_goal"]}
            rows.append(row)
            groups[row["episode_id"]].append(row)
        if len(groups) != 8 or any(len(group) != 4 for group in groups.values()):
            raise ValueError("Episode groups are not eight groups of four")
        summaries = []
        for episode, group in sorted(groups.items()):
            summaries.append({"episode_id": episode, "samples": len(group),
                              "distinct_actions_by_turn": len({encode(r["actions_by_turn"]) for r in group}),
                              "distinct_flattened_actions": len({encode(r["flattened_actions"]) for r in group}),
                              "distinct_terminal_outcomes": len({encode([r["total_reward"], r["task_success"], r["distance_to_goal"]]) for r in group}),
                              "action_counts": [len(r["flattened_actions"]) for r in group]})
        signatures[label] = Counter(encode({"episode_id": r["episode_id"],
                                           "flattened_actions": r["flattened_actions"]}) for r in rows)
        runs.append({"run": label, "configured_seed": run["configured_seed"], "step": 1,
                     "rollout_sha256": run["rollout_sha256"], "episodes": summaries, "records": rows})
    anchor = runs[0]["run"]
    comparisons = [{"reference": anchor, "run": label,
                    "matching_episode_action_sequences": sum((signatures[anchor] & values).values()),
                    "all_32_episode_action_sequences_equal": signatures[anchor] == values}
                   for label, values in signatures.items() if label != anchor]
    report = {"schema": "positive_first_batch_executed_action_diversity_v1",
              "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "sampling_provenance_sha256": digest(args.sampling_provenance),
              "auditor_sha256": digest(Path(__file__)), "runs": runs, "comparisons": comparisons,
              "scope": "Recorded first batches only; executed-action diversity excludes generated text. No live model calls, full-training diversity claim, independent sampling-stream claim, or navigation gain."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runs_checked": len(runs), "comparisons": comparisons,
                      "distinct_flattened_actions_by_episode": [[e["distinct_flattened_actions"] for e in run["episodes"]] for run in runs]}))


if __name__ == "__main__":
    main()
