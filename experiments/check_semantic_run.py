"""Reject EventTrace training logs with missing coverage or reward-service failures."""

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--steps", required=True, type=int)
    args = parser.parse_args()
    lines = [json.loads(line) for line in args.rollout.read_text().splitlines()]
    assert [line["step"] for line in lines] == list(range(1, args.steps + 1))
    infos = [item for line in lines for item in line["info"]]
    assert len(infos) == 8 * args.steps
    verdicts = collections.Counter(
        turn["semantic_verdict"].get("status")
        for item in infos for turn in item["gen_traj"]
    )
    result = {
        "steps": args.steps,
        "rollouts": len(infos),
        "rollouts_with_events": sum(bool(item["semantic_events"]) for item in infos),
        "verdict_counts": dict(verdicts),
        "confirmed_event_hits": sum(
            round(float(item["semantic_progress"]) * len(item["semantic_events"]))
            for item in infos
        ),
    }
    print(json.dumps(result, indent=2))
    assert result["rollouts_with_events"] > 0, "no parsed events"
    assert verdicts["error"] == 0, "verifier or parser service failed"
    assert sum(verdicts.get(key, 0) for key in ("completed", "uncertain")) > 0, \
        "no VLM verdicts reached training"


if __name__ == "__main__":
    main()
