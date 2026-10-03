"""Freeze one first-action success/failure preference per R2R group-four episode.

Only exact same-episode, same-instruction, same-initial-observation pairs
are selected. A fixed hash chooses among valid success/failure variants;
no geodesic intermediate label or val-unseen outcome participates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from prepare_policy_process_manifest import digest
from preflight_group_success_preferences import (SUCCESS, first_divergence,
                                                 valid_preference_info)


FAILURES = {"stopped but goal not reached.",
            "number of turns exceeded."}
SALT = "group4-outcome-first-action-preference-v1|"


def pair_key(seed: int, eid: str, good: int, bad: int) -> str:
    return hashlib.sha256(
        f"{SALT}{seed}:{eid}:{good}:{bad}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollout) != 3:
        raise ValueError("expected three complete group-four source rollouts")
    split = json.loads(args.scene_split.read_text())
    if split["schema"] != "stop_history_lora_scene_split_v1":
        raise ValueError("wrong frozen scene split")
    scene_to_part = {scene: part for part, scenes in
                     split["scene_split"].items() for scene in scenes}
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(row["episode_id"]): row for row in
                    json.load(stream)["episodes"]}
    selected = {part: [] for part in ("fit", "development", "audit")}
    sources = {}
    inventory = Counter()
    for seed, path in zip((11, 22, 33), args.rollout):
        steps = []
        groups = defaultdict(list)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(int(row["step"]))
                for info in row["info"]:
                    groups[str(info["episode_id"])].append(info)
        if steps != list(range(1, 129)) or len(groups) != 512 or \
                any(len(infos) != 4 for infos in groups.values()):
            raise ValueError(f"incomplete group-four seed {seed}")
        sources[str(seed)] = {"path": str(path), "sha256": digest(path),
                              "episode_groups": 512, "rollouts": 2048}
        for eid, infos in groups.items():
            episode = episodes[eid]
            scene = str(episode["scene_id"])
            part = scene_to_part.get(scene)
            if part is None:
                continue
            if any(not valid_preference_info(info) or
                   info["instruction"].strip() !=
                       episode["instruction"]["instruction_text"].strip()
                   for info in infos):
                continue
            inventory[f"{part}_complete_groups"] += 1
            options = []
            for good_variant, good in enumerate(infos):
                if good["end_reason"] != SUCCESS:
                    continue
                for bad_variant, bad in enumerate(infos):
                    if bad["end_reason"] not in FAILURES:
                        continue
                    if first_divergence(good, bad) != 1:
                        continue
                    good_action = [str(action) for action in
                                   good["gen_traj"][0]["executed_actions"]]
                    bad_action = [str(action) for action in
                                  bad["gen_traj"][0]["executed_actions"]]
                    if good_action == bad_action:
                        raise ValueError("identical preferred/rejected action")
                    options.append((pair_key(seed, eid, good_variant,
                                             bad_variant), good_variant,
                                    bad_variant, good_action, bad_action))
            if not options:
                continue
            _, good_variant, bad_variant, good_action, bad_action = min(options)
            row = {"group_id": f"s{seed}_e{eid}", "seed": seed,
                   "episode_id": eid, "scene_id": scene,
                   "trajectory_id": str(episode["trajectory_id"]),
                   "instruction": episode["instruction"]["instruction_text"].strip(),
                   "preferred_variant": good_variant,
                   "rejected_variant": bad_variant,
                   "preferred_response": ", ".join(good_action),
                   "rejected_response": ", ".join(bad_action),
                   "preferred_terminal_mode": SUCCESS,
                   "rejected_terminal_mode": infos[bad_variant]["end_reason"],
                   "shared_prompt_turn": 1}
            selected[part].append(row)
    for part in selected:
        selected[part].sort(key=lambda row: row["group_id"])
        ids = [row["group_id"] for row in selected[part]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate selected group in {part}")
    scenes = {part: {row["scene_id"] for row in rows}
              for part, rows in selected.items()}
    if scenes["fit"] & scenes["development"] or \
            scenes["fit"] & scenes["audit"] or \
            scenes["development"] & scenes["audit"]:
        raise ValueError("scene leakage")
    counts = {part: {"groups": len(rows),
                     "scenes": len(scenes[part]),
                     "unique_episodes": len({row["episode_id"] for row in rows}),
                     "seeds": dict(Counter(row["seed"] for row in rows))}
              for part, rows in selected.items()}
    result = {"schema": "group4_first_action_preference_manifest_v1",
              "selection": "one SHA-ranked success/navigation-failure pair per complete four-rollout group, first divergent action, train scenes only",
              "train_dataset_sha256": digest(args.train_dataset),
              "scene_split_sha256": digest(args.scene_split),
              "sources": sources,
              "inventory": dict(inventory), "counts": counts,
              "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": result["inventory"],
                      "counts": counts, "sha256": digest(args.output)},
                     indent=2))


if __name__ == "__main__":
    main()
