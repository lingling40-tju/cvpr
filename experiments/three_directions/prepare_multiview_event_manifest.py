"""Freeze two-view event replay inputs separately from privileged labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


DATA_SHA = "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea"
BOUNDARY_SHA = "179f726b2f91dd269238559ddc71c35e9c77dbfa1ebd287f51b4e73a7b134b10"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def terminal_clause(instruction: str) -> str:
    clauses = [x.strip() for x in re.split(r"[.!?;]+", instruction)
               if x.strip()]
    if not clauses:
        raise ValueError("empty instruction")
    clause = clauses[-1]
    if len(re.findall(r"[A-Za-z]+", clause)) < 3:
        clause = instruction.strip()
    return " ".join(clause.split()[-40:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--coverage-preflight", type=Path, required=True)
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    if len({args.capture_output, args.labels_output, args.report_output}) != 3 or \
            digest(args.dataset) != DATA_SHA or \
            digest(args.boundary_manifest) != BOUNDARY_SHA:
        raise ValueError("changed source or colliding outputs")
    boundary = json.loads(args.boundary_manifest.read_text())
    coverage = json.loads(args.coverage_preflight.read_text())
    if boundary["schema"] != "boundary_occupancy_rgb_replay_manifest_v1" or \
            boundary["group_size"] != 4 or boundary["seeds"] != [11, 22, 33] or \
            coverage["schema"] != "multiview_event_train_source_preflight_v1" or \
            coverage["manifest_sha256"] != BOUNDARY_SHA:
        raise ValueError("invalid frozen n4 source")
    with gzip.open(args.dataset, "rt", encoding="utf-8") as stream:
        episodes = {str(x["episode_id"]): x for x in
                    json.load(stream)["episodes"]}
    scene_to_part = {scene: part for part in ("fit", "development")
                     for scene in boundary["scene_split"][part]}
    positive_by_key = {
        (x["seed"], str(x["episode_id"]), x["variant"]): x
        for part in ("fit", "development")
        for x in boundary["selected"][part]
    }
    if len(positive_by_key) != 1472:
        raise ValueError("changed crossing positives")
    capture = {part: {} for part in ("fit", "development")}
    labels = {part: [] for part in capture}
    counts = {part: Counter() for part in capture}
    ids = {part: defaultdict(set) for part in capture}
    old_states = {part: set() for part in capture}
    new_states = {part: set() for part in capture}
    seen = set()

    def add(part: str, key: tuple, info: dict, indexes: tuple[int, int],
            kind: str, distances: list[float]) -> None:
        seed, eid, variant = key
        before, after = indexes
        if not 0 <= before < after < len(distances):
            raise ValueError("invalid pair indices")
        record_id = f"s{seed}_e{eid}_v{variant}"
        episode = episodes[eid]
        instruction = episode["instruction"]["instruction_text"].strip()
        entry = capture[part].get(record_id)
        if entry is None:
            entry = {
                "record_id": record_id, "seed": seed, "episode_id": eid,
                "variant": variant, "scene_id": str(episode["scene_id"]),
                "instruction": instruction,
                "terminal_clause": terminal_clause(instruction),
                "wrong_instruction": None,
                "wrong_terminal_clause": None,
                "states_to_capture": [],
            }
            capture[part][record_id] = entry
        if entry["instruction"] != instruction:
            raise ValueError("inconsistent natural instruction")
        for index in indexes:
            if index not in entry["states_to_capture"]:
                entry["states_to_capture"].append(index)
            new_states[part].add((key, index))
        pair_id = record_id + ":" + kind
        if any(x["pair_id"] == pair_id for x in labels[part]):
            raise ValueError("duplicate class on trajectory")
        labels[part].append({
            "pair_id": pair_id, "record_id": record_id,
            "kind": kind, "before_state_index": before,
            "after_state_index": after,
            "before_distance_m_for_audit_only": distances[before],
            "after_distance_m_for_audit_only": distances[after],
            "task_success_for_audit_only": bool(info["task_success"]),
        })
        counts[part][kind] += 1
        ids[part][kind].add(eid)

    for seed in (11, 22, 33):
        rollout = args.root / (
            f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl")
        if digest(rollout) != boundary["source_sha256"]["seeds"][str(seed)][
                "rollout"]:
            raise ValueError(f"audited seed {seed} changed")
        for line in rollout.open():
            batch = json.loads(line)
            groups = defaultdict(list)
            for info in batch["info"]:
                groups[str(info["episode_id"])].append(info)
            if len(groups) != 4 or any(len(v) != 4 for v in groups.values()):
                raise ValueError("not exact n4 group")
            for eid, group in groups.items():
                part = scene_to_part.get(str(episodes[eid]["scene_id"]))
                if part is None:
                    continue  # Audit scenes remain unopened.
                for variant, info in enumerate(group):
                    key = (seed, eid, variant)
                    if key in seen:
                        raise ValueError("repeated rollout identity")
                    seen.add(key)
                    distances = [float(info["oracle_start_distance"])]
                    for turn in info["gen_traj"]:
                        before = float(turn["oracle_before_distance"])
                        after = float(turn["oracle_after_distance"])
                        if not math.isfinite(after) or \
                                abs(before-distances[-1]) > 1e-4:
                            raise ValueError("changed turnwise source distance")
                        distances.append(after)
                    if not distances or abs(distances[-1]-float(
                            info["distance_to_goal"])) > 1e-4:
                        raise ValueError("changed terminal source distance")
                    if info["instruction"].strip() != episodes[eid][
                            "instruction"]["instruction_text"].strip():
                        raise ValueError("changed natural instruction")
                    old = positive_by_key.get(key)
                    if old:
                        outside, inside = old["outside_state_index"], old[
                            "inside_state_index"]
                        if not 3.5 <= distances[outside] <= 4.5 or \
                                not distances[inside] <= 3 or \
                                info["instruction"].strip() != old[
                                    "instruction"]:
                            raise ValueError("changed crossing source")
                        add(part, key, info, (outside, inside), "crossing",
                            distances)
                        old_states[part].update(((key, outside), (key, inside)))
                        far = [(a, a+1) for a in range(max(0, outside-3),
                                                        outside)
                               if distances[a] >= 5 and distances[a+1] >= 5]
                        if far:
                            add(part, key, info, far[-1], "far_nonarrival",
                                distances)
                        if old["wrong_instruction_same_start"]:
                            add(part, key, info, (outside, inside),
                                "wrong_instruction", distances)
                            record = capture[part][f"s{seed}_e{eid}_v{variant}"]
                            record["wrong_instruction"] = old[
                                "wrong_instruction"]
                            record["wrong_terminal_clause"] = terminal_clause(
                                old["wrong_instruction"])
                    retreat = [(a, a+1) for a, b in enumerate(zip(
                        distances, distances[1:])) if
                        2.5 <= b[0] <= 5 and b[1]-b[0] >= .5 and
                        b[1] >= 3.5]
                    if retreat:
                        add(part, key, info, retreat[0], "retreat",
                            distances)
    if len(seen) != 5268:
        raise ValueError("changed fit/development rollout coverage")
    expected = {
        "fit": {"crossing": 1184, "far_nonarrival": 897,
                "wrong_instruction": 644, "retreat": 283},
        "development": {"crossing": 288, "far_nonarrival": 202,
                        "wrong_instruction": 176, "retreat": 92},
    }
    if {part: dict(counter) for part, counter in counts.items()} != expected:
        raise ValueError("new source selection differs from CPU preflight")
    gates = {
        "fit_positive_episode_ids_at_least_150":
            len(ids["fit"]["crossing"]) >= 150,
        "development_positive_episode_ids_at_least_50":
            len(ids["development"]["crossing"]) >= 50,
        "development_far_episode_ids_at_least_50":
            len(ids["development"]["far_nonarrival"]) >= 50,
        "development_retreat_episode_ids_at_least_25":
            len(ids["development"]["retreat"]) >= 25,
        "development_wrong_episode_ids_at_least_30":
            len(ids["development"]["wrong_instruction"]) >= 30,
    }
    if not all(gates.values()):
        raise ValueError("insufficient frozen event source coverage")
    ordered_capture = {part: sorted(items.values(), key=lambda x:
                       (x["scene_id"], x["seed"], str(x["episode_id"]),
                        x["variant"])) for part, items in capture.items()}
    for items in ordered_capture.values():
        for item in items:
            item["states_to_capture"].sort()
    capture_value = {
        "schema": "multiview_event_rgb_capture_manifest_v1",
        "group_size": 4, "seeds": [11, 22, 33],
        "boundary_manifest_sha256": BOUNDARY_SHA,
        "coverage_preflight_sha256": digest(args.coverage_preflight),
        "source_sha256": boundary["source_sha256"],
        "scene_split": boundary["scene_split"],
        "terminal_clause_rule":
            "last punctuation-delimited clause; fall back to full instruction if <3 alphabetic words; keep final <=40 whitespace tokens",
        "selected": ordered_capture,
        "interpretation": "Model input source metadata only; pair classes and geodesic distances are separate",
    }
    save(args.capture_output, capture_value)
    labels_value = {
        "schema": "multiview_event_privileged_pair_labels_v1",
        "group_size": 4,
        "capture_manifest_sha256": digest(args.capture_output),
        "selected": labels,
        "interpretation": "Supervision and replay audit only; never serialize into model prompt",
    }
    save(args.labels_output, labels_value)
    report = {
        "schema": "multiview_event_frozen_source_report_v1",
        "boundary_manifest_sha256": BOUNDARY_SHA,
        "capture_manifest_sha256": digest(args.capture_output),
        "labels_sha256": digest(args.labels_output),
        "counts": {part: dict(counts[part]) for part in counts},
        "unique_episode_ids": {
            part: {kind: len(values) for kind, values in ids[part].items()}
            for part in ids},
        "capture_records": {part: len(ordered_capture[part])
                            for part in ordered_capture},
        "capture_state_count": {
            part: len(new_states[part]) for part in new_states},
        "verified_crossing_states_reusable": {
            part: len(old_states[part]) for part in old_states},
        "extra_states_to_render": {
            part: len(new_states[part]-old_states[part])
            for part in new_states},
        "coverage_gates": gates,
        "ready_for_rgb_replay": all(gates.values()),
        "audit_selected": False,
        "navigation_result": False,
    }
    save(args.report_output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
