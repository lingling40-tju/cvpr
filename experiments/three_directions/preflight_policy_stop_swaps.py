"""Count safe natural wrong-goal instructions for policy STOP histories.

Only train-split metadata is read. A 6.5 m gap between target goals and
a successful endpoint within 3 m of the original goal guarantee at
least 3.5 m Euclidean separation from the wrong goal. This is a data
coverage audit, not a model-quality or navigation result.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from prepare_policy_process_manifest import valid_info


GAPS_M = (3.5, 4.5, 5.5, 6.5)
SUCCESS = "successfully reached the goal."


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pose(row: dict) -> tuple:
    return (str(row["scene_id"]),
            tuple(round(float(v), 4) for v in row["start_position"]),
            tuple(round(float(v), 4) for v in row["start_rotation"]))


def has_alternative(row: dict, by_pose: dict, threshold: float) -> bool:
    target = row["goals"][0]["position"]
    return any(str(other["trajectory_id"]) != str(row["trajectory_id"]) and
               math.dist(target, other["goals"][0]["position"]) >= threshold and
               other["instruction"]["instruction_text"].strip() !=
               row["instruction"]["instruction_text"].strip()
               for other in by_pose[pose(row)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--scene-split", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    dataset_sha = digest(args.dataset)
    if manifest["schema"] != "policy_process_train_manifest_v1" or \
            manifest["train_dataset_sha256"] != dataset_sha:
        raise ValueError("policy manifest or R2R-train source mismatch")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["episodes"]
    by_id = {str(row["episode_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate R2R-train episode ID")
    by_pose = defaultdict(list)
    for row in rows:
        by_pose[pose(row)].append(row)

    safe_ids = {threshold: {str(row["episode_id"]) for row in rows
                            if has_alternative(row, by_pose, threshold)}
                for threshold in GAPS_M}

    parts = {}
    scenes = {}
    for part, plans in manifest["selected"].items():
        expected = manifest["targets"][part]
        if len(plans) != expected:
            raise ValueError(f"{part} selected count changed")
        scenes[part] = {str(plan["scene_id"]) for plan in plans}
        successes = [plan for plan in plans if plan["terminal_mode"] == SUCCESS]
        coverage = {}
        for threshold in GAPS_M:
            selected = []
            for plan in successes:
                eid = str(plan["episode_id"])
                row = by_id[eid]
                source_scene = str(row["scene_id"])
                prefix = "data/scene_datasets/"
                if source_scene.startswith(prefix):
                    source_scene = source_scene[len(prefix):]
                if str(plan["scene_id"]) != source_scene:
                    raise ValueError("policy plan scene mismatch")
                record_id = f"s{plan['seed']}_e{eid}_v{plan['variant']}"
                path = args.records_root / part / "records" / f"{record_id}.json"
                record = json.loads(path.read_text())
                if record["record_id"] != record_id or \
                        record["terminal_mode"] != SUCCESS or \
                        str(record["episode_id"]) != eid:
                    raise ValueError("policy record identity mismatch")
                if eid in safe_ids[threshold]:
                    selected.append(plan)
            coverage[str(threshold)] = {
                "records": len(selected),
                "unique_episodes": len({str(x["episode_id"]) for x in selected}),
                "scenes": len({str(x["scene_id"]) for x in selected}),
            }
        parts[part] = {"selected_records": len(plans),
                       "successful_records": len(successes),
                       "safe_swap_by_goal_gap_m": coverage}
    for left, right in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit")):
        if scenes[left] & scenes[right]:
            raise ValueError("scene partition leakage")
    report = {
        "schema": "policy_stop_swap_coverage_v2",
        "source_sha256": {"dataset": dataset_sha,
                          "manifest": digest(args.manifest)},
        "parts": parts,
        "interpretation": (
            "Train-scene metadata and cached policy-history coverage only. "
            "The 6.5m goal-gap rule guarantees a wrong-goal endpoint at "
            "least 3.5m away for a successful original STOP within 3m; "
            "smaller gaps do not guarantee this without endpoint replay. "
            "No model fit or navigation gain is implied."
        ),
    }
    if args.source_root or args.scene_split:
        if not args.source_root or not args.scene_split:
            raise ValueError("source-root and scene-split must be supplied together")
        if digest(args.scene_split) != manifest["scene_split_sha256"]:
            raise ValueError("frozen scene split hash mismatch")
        split = json.loads(args.scene_split.read_text())
        scene_to_part = {scene: part
                         for part, scene_list in split["scene_split"].items()
                         for scene in scene_list}
        if len(scene_to_part) != sum(map(len, split["scene_split"].values())):
            raise ValueError("overlapping scene partitions")
        all_source = {part: {str(t): {"records": 0, "episodes": set(),
                                    "scenes": set()} for t in GAPS_M}
                      for part in split["scene_split"]}
        success_counts = {part: 0 for part in all_source}
        for seed, source in manifest["sources"].items():
            path = args.source_root / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError(f"source rollout hash mismatch for seed {seed}")
            steps, count = [], 0
            with path.open() as stream:
                for line in stream:
                    step = json.loads(line)
                    steps.append(int(step["step"]))
                    for info in step["info"]:
                        count += 1
                        eid = str(info["episode_id"])
                        row = by_id[eid]
                        part = scene_to_part.get(str(row["scene_id"]))
                        if part is None or not valid_info(info) or \
                                info["instruction"].strip() != \
                                row["instruction"]["instruction_text"].strip() or \
                                info["end_reason"] != SUCCESS:
                            continue
                        success_counts[part] += 1
                        for threshold in GAPS_M:
                            if eid in safe_ids[threshold]:
                                cell = all_source[part][str(threshold)]
                                cell["records"] += 1
                                cell["episodes"].add(eid)
                                cell["scenes"].add(str(row["scene_id"]))
            if steps != list(range(1, 129)) or count != source["rollouts"]:
                raise ValueError(f"source rollout incomplete for seed {seed}")
        report["all_source_eligible"] = {
            part: {"successful_records": success_counts[part],
                   "safe_swap_by_goal_gap_m": {
                       gap: {"records": cell["records"],
                             "unique_episodes": len(cell["episodes"]),
                             "scenes": len(cell["scenes"])}
                       for gap, cell in gaps.items()}}
            for part, gaps in all_source.items()}
        report["source_sha256"]["scene_split"] = digest(args.scene_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
