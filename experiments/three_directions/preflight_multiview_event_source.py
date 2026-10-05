"""CPU-only source coverage for a distinct two-view event reward hypothesis."""

import collections
import hashlib
import json
from pathlib import Path
import sys


root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert manifest_sha == "179f726b2f91dd269238559ddc71c35e9c77dbfa1ebd287f51b4e73a7b134b10"
manifest = json.loads(manifest_path.read_text())
assert manifest["group_size"] == 4 and manifest["seeds"] == [11, 22, 33]
plans = {(p["seed"], str(p["episode_id"]), p["variant"]): (part, p)
         for part in ("fit", "development") for p in manifest["selected"][part]}
assert len(plans) == 1472
stats = {part: collections.Counter() for part in ("fit", "development")}
ids = {part: collections.defaultdict(set) for part in stats}
selected_frames = {part: set() for part in stats}
old_frames = {part: set() for part in stats}
seen = set()
for seed in (11, 22, 33):
    rollout = root / f"verl_checkpoints/oracle_turnwise_exact512_128_seed{seed}/rollout.jsonl"
    digest = hashlib.sha256(rollout.read_bytes()).hexdigest()
    assert digest == manifest["source_sha256"]["seeds"][str(seed)]["rollout"]
    for line in rollout.open():
        batch = json.loads(line)
        groups = collections.defaultdict(list)
        for info in batch["info"]:
            groups[str(info["episode_id"])].append(info)
        assert len(groups) == 4 and all(len(x) == 4 for x in groups.values())
        for eid, group in groups.items():
            for variant, info in enumerate(group):
                key = (seed, eid, variant)
                if key not in plans:
                    continue
                assert key not in seen
                seen.add(key)
                part, p = plans[key]
                distances = [float(info["oracle_start_distance"])]
                for turn in info["gen_traj"]:
                    assert abs(float(turn["oracle_before_distance"]) - distances[-1]) <= 1e-4
                    distances.append(float(turn["oracle_after_distance"]))
                assert abs(distances[-1] - float(info["distance_to_goal"])) <= 1e-4
                outside, inside = p["outside_state_index"], p["inside_state_index"]
                assert 0 <= outside < inside <= outside+2 < len(distances)+2
                assert 3.5 <= distances[outside] <= 4.5 and distances[inside] <= 3
                stats[part]["records"] += 1
                stats[part]["near_failure_records"] += not info["task_success"]
                ids[part]["all"].add(eid)
                if not info["task_success"]:
                    ids[part]["near_failure"].add(eid)
                if p["wrong_instruction_same_start"]:
                    ids[part]["exact_wrong"].add(eid)
                windows = [range(max(0, target-3), target+1)
                           for target in (outside, inside)]
                for window in windows:
                    for state in window:
                        selected_frames[part].add((key, state))
                for state in (outside, inside):
                    old_frames[part].add((key, state))
                stats[part]["both_full_four_state_windows"] += \
                    outside >= 3 and inside >= 3
                stats[part]["crossing_one_turn"] += inside-outside == 1
                stats[part]["crossing_two_turns"] += inside-outside == 2
                # Negative event pairs stay within the same visual context.
                near_window = sorted(set(windows[0]) | set(windows[1]))
                far_pair = any(a+1 == b and distances[a] >= 5 and
                               distances[b] >= 5 for a, b in
                               zip(near_window, near_window[1:]))
                stall_pair = any(a+1 == b and distances[a] >= 3.5 and
                                 distances[b] >= 3.5 and
                                 abs(distances[b]-distances[a]) <= .1
                                 for a, b in zip(near_window,
                                                 near_window[1:]))
                retreat_pair = any(a+1 == b and distances[a] >= 3.5 and
                                   distances[b]-distances[a] >= .5
                                   for a, b in zip(near_window,
                                                   near_window[1:]))
                stats[part]["far_nonarrival_context"] += far_pair
                stats[part]["stalled_outside_context"] += stall_pair
                stats[part]["retreat_outside_context"] += retreat_pair
                if far_pair or stall_pair or retreat_pair:
                    ids[part]["has_nonarrival_context"].add(eid)
assert len(seen) == len(plans)
result = {
    "schema": "multiview_event_train_source_preflight_v1",
    "manifest_sha256": manifest_sha,
    "parts": {part: {
        **dict(stats[part]),
        "unique_episode_ids": {key: len(value) for key, value in ids[part].items()},
        "old_frames": len(old_frames[part]),
        "multiview_frames": len(selected_frames[part]),
        "additional_frames": len(selected_frames[part]-old_frames[part]),
    } for part in stats},
    "interpretation": "CPU-only source coverage; not model accuracy or navigation",
}
print(json.dumps(result, indent=2))
