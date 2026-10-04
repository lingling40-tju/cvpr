"""Check the frozen same-start development gate without model inference.

The hard-pair guard was added to NEXT_PROCESS_REWARD_PROTOCOL.md in
commit c2eebb7 before the first development checkpoint was reported.
This script consumes the fitted model's aggregate development report;
it cannot substitute for the prospective scene audit or navigation test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ACTION_BASELINE = {"3": 0.6827925925925926, "6": 0.6750902739820072}
HARD_EPISODE_GROUPS = {"3": 26, "6": 36}


def check(report: dict) -> dict:
    if report.get("schema") != "same_start_relative_lora_development_v1":
        raise ValueError("unexpected development report schema")
    selected_step = report["selected_step"]
    selected = report["selected_development"]
    selected_rows = [row for row in report["history"]
                     if row["step"] == selected_step]
    if len(selected_rows) != 1 or selected_rows[0]["development"] != selected:
        raise ValueError("selected checkpoint does not match history")
    if selected_rows[0]["gate"] != report["development_gate"]:
        raise ValueError("stored gate differs from selected history")

    checks = {"instruction_grounding":
              selected["instruction_gain_preference"] >= .75,
              "instruction_pairs_48": selected["instruction_pairs"] == 48}
    for anchor in ("3", "6"):
        row = selected["anchors"][anchor]
        episode = row["episode_macro_accuracy"]
        scene = row["scene_macro_accuracy"]
        hard = row["hard_episode_macro_accuracy"]
        for value in (episode, scene, hard):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid anchor {anchor} accuracy: {value}")
        checks[f"anchor{anchor}_episode_macro"] = episode >= .75
        checks[f"anchor{anchor}_over_action_baseline"] = (
            episode - ACTION_BASELINE[anchor] >= .05)
        checks[f"anchor{anchor}_scene_macro"] = scene >= .70
        checks[f"anchor{anchor}_hard_episode_macro"] = hard >= .60
        checks[f"anchor{anchor}_hard_group_count"] = (
            row["hard_episode_groups"] == HARD_EPISODE_GROUPS[anchor])

    existing = {name: value for name, value in checks.items()
                if name in report["development_gate"]}
    if existing != report["development_gate"]:
        raise ValueError("original frozen gates do not recompute")
    return {
        "schema": "same_start_relative_complete_development_gate_v1",
        "selected_step": selected_step,
        "checks": checks,
        "all_passed": all(checks.values()),
        "interpretation": "Reused R2R-train development only; no prospective audit or navigation result",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.report.read_text()))
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
