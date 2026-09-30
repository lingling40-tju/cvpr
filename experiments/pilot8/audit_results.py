"""Recompute manuscript pilot counts from the original rollout files."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(arm):
    data = []
    for line in (ROOT / f"{arm}_rollout.jsonl").read_text().splitlines():
        row = json.loads(line)
        data.extend((row["step"], info) for info in row["info"])
    return data


def report(rows):
    infos = [info for _, info in rows]
    return {
        "rollouts": len(infos),
        "successes": sum(bool(info["task_success"]) for info in infos),
        "mean_return": sum(float(info["total_reward"]) for info in infos) / len(infos),
        "mean_event_progress": sum(float(info.get("semantic_progress", 0)) for info in infos) / len(infos),
        "event_hits": sum(sum(turn.get("semantic_verdict", {}).get("status") == "completed"
                              for turn in info["gen_traj"]) for info in infos),
        "uncertain_turns": sum(sum(turn.get("semantic_verdict", {}).get("status") == "uncertain"
                                   for turn in info["gen_traj"]) for info in infos),
    }


def main():
    control, event = load("control"), load("event")
    summary = json.loads((ROOT / "summary.json").read_text())
    for arm, rows in (("control", control), ("event", event)):
        got = report(rows)
        expected = summary[arm]
        assert got["rollouts"] == expected["rollouts"] == 16
        assert got["successes"] == expected["successes"] == 13
        assert abs(got["mean_return"] - expected["mean_total_reward"]) < 1e-9
        assert abs(got["mean_event_progress"] - expected["mean_event_progress"]) < 1e-9
        assert got["event_hits"] == expected["event_hits"]
        assert got["uncertain_turns"] == expected["uncertain_turns"]
        print(arm, got)

    paired = 0
    for (step_c, c), (step_e, e) in zip(control, event):
        if step_c != 1:
            continue
        assert step_e == 1 and c["episode_id"] == e["episode_id"]
        assert [t["response"] for t in c["gen_traj"]] == [t["response"] for t in e["gen_traj"]]
        assert abs((e["total_reward"] - c["total_reward"]) - e["semantic_progress"]) < 1e-6
        paired += 1
    assert paired == 8
    print("matched_first_step_rollouts", paired)


if __name__ == "__main__":
    main()
