"""Count train-only rollout endpoints usable for a STOP-readiness screen.

No RGB is read and no checkpoint is fitted. The result is a data-availability
audit, not a semantic-verifier accuracy or navigation result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def label(info: dict) -> str:
    distance = float(info["distance_to_goal"])
    if info["task_success"] and distance <= 3.0:
        return "positive_success_stop"
    if info["end_reason"] == "stopped but goal not reached." and distance >= 3.5:
        return "negative_failed_stop"
    if not info["task_success"] and distance < 3.5:
        return "ambiguous_near_goal_failure"
    return "other_failure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    scenes = {str(row["episode_id"]): str(row["scene_id"]) for row in episodes}
    all_counts = Counter()
    end_reasons: dict[str, Counter] = defaultdict(Counter)
    scene_counts: dict[str, Counter] = defaultdict(Counter)
    episode_ids_by_label: dict[str, set[str]] = defaultdict(set)
    scene_episode_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    by_rollout = {}
    for path in args.rollout:
        steps, groups = [], defaultdict(list)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(int(row["step"]))
                for info in row["info"]:
                    episode_id = str(info["episode_id"])
                    if episode_id not in scenes or info["data_source"] != "r2r":
                        raise ValueError(f"unexpected episode {episode_id}")
                    groups[episode_id].append(info)
                    key = label(info)
                    all_counts[key] += 1
                    end_reasons[key][str(info["end_reason"])] += 1
                    scene_counts[scenes[episode_id]][key] += 1
                    episode_ids_by_label[key].add(episode_id)
                    scene_episode_ids[scenes[episode_id]][key].add(episode_id)
        if steps != list(range(1, 129)) or len(groups) != 512 or any(len(v) != 4 for v in groups.values()):
            raise ValueError(f"incomplete group-four rollout {path}")
        local = Counter(label(info) for values in groups.values() for info in values)
        by_rollout[path.parent.name] = {
            "source_sha256": digest(path),
            "groups": len(groups), "rollouts": sum(map(len, groups.values())),
            "labels": dict(local),
        }
    result = {
        "schema": "train_only_stop_supervision_preflight_v1",
        "train_dataset_sha256": digest(args.train_dataset),
        "label_rule": "success within 3.0m is positive; failed voluntary STOP at least 3.5m away is negative; near-goal failures are ambiguous",
        "sources": by_rollout,
        "total_labels": dict(all_counts),
        "unique_episodes_by_label": {key: len(ids) for key, ids in episode_ids_by_label.items()},
        "end_reasons_by_label": {key: dict(counts) for key, counts in end_reasons.items()},
        "scene_counts": {scene: dict(counts) for scene, counts in sorted(scene_counts.items())},
        "scene_unique_episodes_by_label": {
            scene: {key: len(ids) for key, ids in sorted(labels.items())}
            for scene, labels in sorted(scene_episode_ids.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sources": by_rollout,
                      "total_labels": dict(all_counts),
                      "unique_episodes_by_label": result["unique_episodes_by_label"],
                      "end_reasons_by_label": result["end_reasons_by_label"],
                      "scenes": len(scene_counts)}, indent=2))


if __name__ == "__main__":
    main()
