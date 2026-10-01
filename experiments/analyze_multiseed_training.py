"""Summarize raw 64-step, three-seed matched ActiveVLN training rollouts."""

import collections
import json
from pathlib import Path


ROOT = Path("/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930")
STEPS = 64
SEEDS = (11, 22, 33)


def read_arm(seed, arm):
    experiment = f"eventtrace_r2r64_seed{seed}_{arm}"
    path = ROOT / "verl_checkpoints" / experiment / "rollout.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["step"] for line in lines] == list(range(1, STEPS + 1))
    episodes_by_step = []
    infos = []
    diverse_groups = 0
    return_variance_groups = 0
    for line in lines:
        batch = line["info"]
        assert len(batch) == 8
        counts = collections.Counter(str(item["episode_id"]) for item in batch)
        assert len(counts) == 4 and set(counts.values()) == {2}
        episodes_by_step.append(sorted(counts))
        infos.extend(batch)
        groups = collections.defaultdict(list)
        for item in batch:
            groups[str(item["episode_id"])].append(item)
        for pair in groups.values():
            signatures = [tuple(turn["response"] for turn in item["gen_traj"])
                          for item in pair]
            diverse_groups += signatures[0] != signatures[1]
            return_variance_groups += pair[0]["total_reward"] != pair[1]["total_reward"]
    assert len({ep for step in episodes_by_step for ep in step}) == 256
    assert diverse_groups > 0, "all GRPO groups generated identical trajectories"
    assert return_variance_groups > 0, "all GRPO groups have zero return variance"
    successes = sum(bool(item["task_success"]) for item in infos)
    result = {
        "steps": STEPS,
        "train_episodes": 256,
        "rollouts": len(infos),
        "successful_rollouts": successes,
        "rollout_sr": successes / len(infos),
        "diverse_grpo_groups": diverse_groups,
        "nonzero_return_variance_groups": return_variance_groups,
        "mean_return": sum(float(item["total_reward"]) for item in infos) / len(infos),
        "episode_ids_by_step": episodes_by_step,
    }
    if arm == "event":
        verdicts = collections.Counter(turn["semantic_verdict"].get("status")
                                       for item in infos for turn in item["gen_traj"])
        event_types = collections.Counter(event["type"] for item in infos
                                          for event in item["semantic_events"])
        hits = sum(round(float(item["semantic_progress"]) * len(item["semantic_events"]))
                   for item in infos)
        result.update({
            "parser_status_logged": False,
            "rollouts_with_events": sum(bool(item["semantic_events"]) for item in infos),
            "parsed_event_mentions": sum(len(item["semantic_events"]) for item in infos),
            "event_type_mentions": dict(event_types),
            "confirmed_event_hits": hits,
            "verdict_turn_counts": dict(verdicts),
        })
    return result


def main():
    output = {"budget": "64 steps x 4 episodes x 2 rollouts per arm and seed", "seeds": {}}
    for seed in SEEDS:
        control = read_arm(seed, "control")
        event = read_arm(seed, "event")
        assert control["episode_ids_by_step"] == event["episode_ids_by_step"], seed
        for result in (control, event):
            del result["episode_ids_by_step"]
        output["seeds"][str(seed)] = {"control": control, "event": event,
                                       "matched_episode_order": True}
    destination = ROOT / "runlogs" / "eventtrace_r2r64" / "analysis.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
