"""Count exact shared-prompt action preferences in existing group-four rollouts.

This is a data-availability preflight. It does not train a policy or use
val-unseen results. A successful and failed rollout can form a clean
action preference only at the first differing response after identical
earlier response/action history for the same train episode.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from prepare_policy_process_manifest import VALID, digest


SUCCESS = "successfully reached the goal."


def valid_preference_info(info: dict) -> bool:
    if (info.get("data_source") != "r2r" or
            info.get("action_space") != "r2r" or
            info.get("global_start_step") != 1):
        return False
    turns = info.get("gen_traj", [])
    if not turns or len(turns) > 13:
        return False
    actions = [str(action) for turn in turns
               for action in turn.get("executed_actions", [])]
    if not actions or any(action not in VALID for action in actions) or \
            "stop" in actions[:-1]:
        return False
    if len(turns) == 13 and not (
            info.get("end_reason") == "number of turns exceeded." and
            turns[-1].get("executed_actions") == []):
        return False
    return True


def first_divergence(success: dict, failure: dict) -> int | None:
    good = [turn for turn in success["gen_traj"]
            if turn.get("executed_actions")]
    bad = [turn for turn in failure["gen_traj"]
           if turn.get("executed_actions")]
    for index, (left, right) in enumerate(zip(good, bad), 1):
        if left["response"] != right["response"]:
            if left["executed_actions"] != right["executed_actions"]:
                return index
            # Different text is included in the next prompt, so later
            # turns do not share an exact language/visual history.
            return None
        if left["executed_actions"] != right["executed_actions"]:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout) != 3:
        raise ValueError("expected three group-four rollout sources")
    split = json.loads(args.scene_split.read_text())
    scene_to_part = {scene: part for part, scenes in
                     split["scene_split"].items() for scene in scenes}
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(row["episode_id"]): row for row in
                    json.load(stream)["episodes"]}
    summary = {part: {"success_count_per_group": Counter(),
                      "mixed_groups": 0, "shared_prompt_groups": 0,
                      "shared_prompt_pairs": 0,
                      "first_divergence_turn": Counter(),
                      "scenes": set(), "episodes": set()}
               for part in ("fit", "development", "audit", "outside")}
    sources = {}
    for seed, path in zip((11, 22, 33), args.rollout):
        groups = defaultdict(list)
        steps = []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(int(row["step"]))
                for info in row["info"]:
                    groups[str(info["episode_id"])].append(info)
        if steps != list(range(1, 129)) or len(groups) != 512 or \
                any(len(infos) != 4 for infos in groups.values()):
            raise ValueError(f"incomplete group-four source seed {seed}")
        sources[str(seed)] = {"path": str(path), "sha256": digest(path),
                              "groups": 512}
        for eid, infos in groups.items():
            episode = episodes[eid]
            scene = str(episode["scene_id"])
            part = scene_to_part.get(scene, "outside")
            bucket = summary[part]
            bucket["scenes"].add(scene)
            bucket["episodes"].add(eid)
            valid = [info for info in infos if valid_preference_info(info) and
                     info["instruction"].strip() ==
                         episode["instruction"]["instruction_text"].strip()]
            if len(valid) != 4:
                continue
            good = [info for info in valid if info["end_reason"] == SUCCESS]
            bad = [info for info in valid if info["end_reason"] != SUCCESS]
            bucket["success_count_per_group"][len(good)] += 1
            if not good or not bad:
                continue
            bucket["mixed_groups"] += 1
            pair_turns = [turn for success in good for failure in bad
                          if (turn := first_divergence(success, failure))
                          is not None]
            if pair_turns:
                bucket["shared_prompt_groups"] += 1
                bucket["shared_prompt_pairs"] += len(pair_turns)
                bucket["first_divergence_turn"].update(pair_turns)
    output = {"schema": "group4_success_preference_preflight_v1",
              "train_dataset_sha256": digest(args.train_dataset),
              "scene_split_sha256": digest(args.scene_split),
              "sources": sources,
              "parts": {part: {"success_count_per_group":
                                 dict(row["success_count_per_group"]),
                               "mixed_groups": row["mixed_groups"],
                               "shared_prompt_groups":
                                   row["shared_prompt_groups"],
                               "shared_prompt_pairs":
                                   row["shared_prompt_pairs"],
                               "first_divergence_turn":
                                   dict(row["first_divergence_turn"]),
                               "scenes": len(row["scenes"]),
                               "unique_episodes": len(row["episodes"])}
                        for part, row in summary.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
