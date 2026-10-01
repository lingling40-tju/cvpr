"""Summarize corrected 256-step matched runs with the DashScope event judge."""

import collections
import json
from pathlib import Path


ROOT = Path("/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930")
RUN_DIR = ROOT / "runlogs/eventtrace_qwen38_r2r256"
PREFIX = "eventtrace_qwen38_r2r256"
STEPS = 256


def read_arm(seed, arm):
    label = f"seed{seed}_{arm}"
    path = ROOT / "verl_checkpoints" / f"{PREFIX}_{label}" / "rollout.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["step"] for line in lines] == list(range(1, STEPS + 1))
    infos = []
    episodes_by_step = []
    groups = collections.defaultdict(list)
    for line in lines:
        batch = line["info"]
        assert len(batch) == 8
        counts = collections.Counter(str(item["episode_id"]) for item in batch)
        assert len(counts) == 4 and set(counts.values()) == {2}
        episodes_by_step.append(sorted(counts))
        infos.extend(batch)
        for item in batch:
            groups[(line["step"], str(item["episode_id"]))].append(item)
    assert len(groups) == 4 * STEPS
    diverse = 0
    return_variance = 0
    for pair in groups.values():
        assert len(pair) == 2
        signatures = [tuple(turn["response"] for turn in item["gen_traj"])
                      for item in pair]
        diverse += signatures[0] != signatures[1]
        return_variance += pair[0]["total_reward"] != pair[1]["total_reward"]
    successes = sum(bool(item["task_success"]) for item in infos)
    result = {
        "steps": STEPS,
        "train_episode_exposures": 4 * STEPS,
        "unique_train_episodes": len({episode for step in episodes_by_step for episode in step}),
        "rollouts": len(infos),
        "successful_rollouts": successes,
        "rollout_sr": successes / len(infos),
        "diverse_grpo_groups": diverse,
        "nonzero_return_variance_groups": return_variance,
        "mean_return": sum(float(item["total_reward"]) for item in infos) / len(infos),
        "episode_ids_by_step": episodes_by_step,
    }
    if arm == "event":
        verdicts = collections.Counter(turn["semantic_verdict"].get("status")
                                       for item in infos for turn in item["gen_traj"])
        assert verdicts["error"] == 0
        result.update({
            "rollouts_with_events": sum(bool(item["semantic_events"]) for item in infos),
            "parsed_event_mentions": sum(len(item["semantic_events"]) for item in infos),
            "confirmed_event_hits": sum(
                round(float(item["semantic_progress"]) * len(item["semantic_events"]))
                for item in infos
            ),
            "verdict_turn_counts": dict(verdicts),
        })
    before = json.loads((RUN_DIR / f"{label}.proxy_before.json").read_text())
    after = json.loads((RUN_DIR / f"{label}.proxy_after.json").read_text())
    assert before["model"] == after["model"] == "qwen3.8-max-0902"
    assert before["strict_transition"] == after["strict_transition"]
    assert before["reasoning_effort"] == after["reasoning_effort"]
    result["proxy_counter_delta"] = {
        key: after[key] - before[key] for key, value in before.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    assert result["proxy_counter_delta"]["parser_errors"] == 0
    assert result["proxy_counter_delta"]["malformed_responses"] == 0
    if arm == "event":
        assert result["proxy_counter_delta"]["api_calls"] > 0
    return result


def main():
    output = {
        "budget": "256 steps x 4 train episodes x 2 rollouts per arm and seed",
        "event_parser": "local Qwen3-VL-8B-Instruct (held constant)",
        "event_verifier": "DashScope qwen3.8-max-0902, non-thinking, temperature 0",
        "seeds": {},
    }
    for seed in (11, 22, 33):
        control = read_arm(seed, "control")
        event = read_arm(seed, "event")
        assert control["episode_ids_by_step"] == event["episode_ids_by_step"]
        for result in (control, event):
            del result["episode_ids_by_step"]
        output["seeds"][str(seed)] = {
            "control": control, "event": event, "matched_episode_order": True,
        }
    (RUN_DIR / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
