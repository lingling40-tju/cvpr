#!/usr/bin/env python3
"""Record source settings and first-batch signatures for the frozen sampler."""

import argparse
import ast
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    sources = ["tools/run_positive_scale_train.sh",
               "verl/workers/agent/parallel_env_vlnce.py",
               "verl/workers/sharding_manager/fsdp_vllm.py",
               "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
               "verl/workers/actor/dp_actor.py"]
    require("agent_sampling_params.seed = None" in (args.root / sources[1]).read_text(),
            "This audit describes the frozen seed-clearing agent only")
    require("manual_seed(gen_dp_rank + 1000)" in (args.root / sources[2]).read_text(),
            "This audit describes the frozen common-initial-state manager only")
    observed = []
    signatures_by_run = {}
    for seed in (11, 22, 33):
        for arm in ("control", "candidate"):
            label = "positive_trajectory_{}_128step_seed{}".format(arm, seed)
            run = args.root / "runlogs" / label
            if not (run / "completed").is_file():
                continue
            require(not (run / "failed").exists(), "Completed run also marked failed")
            with (run / "train.log").open(errors="replace") as stream:
                prefix = stream.read(1_500_000)
            kwargs = []
            for line in prefix.splitlines():
                if "kwargs: {" in line:
                    value = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line.split("kwargs: ", 1)[1])
                    try:
                        parsed = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        continue
                    kwargs.append({key: parsed[key] for key in
                                   ("seed", "temperature", "top_p", "n", "max_tokens") if key in parsed})
            require(kwargs and all(item.get("seed") == seed for item in kwargs),
                    "Logged standard SamplingParams seed differs from run label")
            path = args.root / "verl_checkpoints" / label / "rollout.jsonl"
            with path.open() as stream:
                first = json.loads(next(stream))
            require(first["step"] == 1 and len(first["info"]) == 32, "First batch differs")
            signatures = []
            per_episode = defaultdict(list)
            for item in first["info"]:
                signature = {"episode_id": str(item["episode_id"]),
                             "turns": [{"response": turn.get("response"),
                                        "executed_actions": turn.get("executed_actions")}
                                       for turn in item["gen_traj"]],
                             "total_reward": item["total_reward"], "task_success": item["task_success"],
                             "end_reason": item["end_reason"], "distance_to_goal": item["distance_to_goal"]}
                encoded = json.dumps(signature, sort_keys=True, allow_nan=False)
                signatures.append(encoded)
                per_episode[signature["episode_id"]].append(encoded)
            require(len(per_episode) == 8 and all(len(items) == 4 for items in per_episode.values()),
                    "First batch episode groups differ")
            signatures_by_run[label] = Counter(signatures)
            observed.append({"run": label, "configured_seed": seed, "completed": (run / "completed").read_text().strip(),
                             "logged_standard_sampling_kwargs": kwargs,
                             "rollout_sha256": digest(path), "first_step": 1, "first_batch_trajectories": 32,
                             "signature_sha256": hashlib.sha256(json.dumps(sorted(signatures)).encode()).hexdigest(),
                             "distinct_signatures_per_episode": {key: len(set(items)) for key, items in sorted(per_episode.items())}})
    require(bool(observed), "No completed runs available")
    anchor = "positive_trajectory_control_128step_seed11"
    require(anchor in signatures_by_run, "Reference first batch unavailable")
    comparisons = [{"reference": anchor, "run": label,
                    "exact_signature_matches": sum((signatures_by_run[anchor] & counts).values()),
                    "all_32_signatures_equal": counts == signatures_by_run[anchor]}
                   for label, counts in sorted(signatures_by_run.items()) if label != anchor]
    report = {"schema": "positive_sampling_seed_provenance_v1",
              "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": {name: digest(args.root / name) for name in sources},
              "auditor_sha256": digest(Path(__file__)), "checked_runs": observed,
              "first_batch_comparisons": comparisons,
              "source_agent_request_seed": None, "source_initial_gpu_generation_seed_rule": "1000 + DP rank",
              "signature_definition": "Sorted JSON of episode ID, per-turn response/executed actions, total reward, task success, end reason, and final distance; elapsed-time fields excluded",
              "scope": "Source settings, logged constructor kwargs, and recorded first batch; no live CUDA RNG-state introspection, whole-training identity check, or attribution of later run variance",
              "interpretation": "Configured data/engine seeds are passed, while the agent clears request seed and the sharding manager defines a common initial GPU sampling rule. Independently varying all rollout RNG streams is not established."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runs_checked": len(observed), "first_batch_comparisons": comparisons,
                      "source_initial_gpu_generation_seed_rule": report["source_initial_gpu_generation_seed_rule"]}))


if __name__ == "__main__":
    main()
