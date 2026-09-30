"""Summarize the two short ActiveVLN runs without claiming statistical gain."""

import json
import statistics
from pathlib import Path


ROOT = Path("/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930")


def summarize(arm):
    path = ROOT / "verl_checkpoints" / f"semantic_pilot8_{arm}_20260930" / "rollout.jsonl"
    rows = []
    if not path.exists():
        return {"error": f"missing {path}"}
    for line in path.read_text().splitlines():
        batch = json.loads(line)
        for info in batch["info"]:
            if not isinstance(info, dict) or "task_success" not in info:
                continue
            turns = info.get("gen_traj", [])
            labels = [t.get("semantic_verdict", {}).get("status") for t in turns]
            rows.append({
                "step": batch["step"],
                "episode_id": info.get("episode_id"),
                "success": bool(info["task_success"]),
                "reward": float(info.get("total_reward", 0)),
                "distance_to_goal": float(info.get("distance_to_goal", 0)),
                "event_progress": float(info.get("semantic_progress", 0)),
                "event_count": len(info.get("semantic_events", [])),
                "event_hits": labels.count("completed"),
                "uncertain_turns": labels.count("uncertain"),
                "error_turns": labels.count("error"),
                "turn_count": len(turns),
            })
    if not rows:
        return {"error": "no finished rollouts"}
    return {
        "arm": arm,
        "rollouts": len(rows),
        "episodes": len(set(row["episode_id"] for row in rows)),
        "steps": sorted(set(row["step"] for row in rows)),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "mean_total_reward": statistics.mean(row["reward"] for row in rows),
        "mean_event_progress": statistics.mean(row["event_progress"] for row in rows),
        "event_hits": sum(row["event_hits"] for row in rows),
        "uncertain_turns": sum(row["uncertain_turns"] for row in rows),
        "error_turns": sum(row["error_turns"] for row in rows),
        "rows": rows,
    }


def main():
    result = {arm: summarize(arm) for arm in ("control", "event")}
    output = ROOT / "runlogs" / "semantic_pilot8_summary.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({arm: {k: v for k, v in data.items() if k != "rows"}
                      for arm, data in result.items()}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
