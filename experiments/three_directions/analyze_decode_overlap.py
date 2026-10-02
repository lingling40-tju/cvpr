"""Compare two separate decodes of the same fixed 256 val-unseen episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_scaled_val256 import load_label


SCREEN_SHA = "546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46"
FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"


def ids_and_scenes(path: Path, sha: str) -> tuple[list[str], list[str]]:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    data = json.loads(path.read_text())
    ids = [str(item) for item in data["episode_ids"]]
    scenes = [str(item) for item in data["scene_ids"]]
    assert len(ids) == len(set(ids)) == len(scenes)
    return ids, scenes


def metric(rows: dict, ids: list[str], key: str) -> float:
    if key == "success":
        return sum(bool(rows[eid][key]) for eid in ids) / len(ids)
    return sum(float(rows[eid][key]) for eid in ids) / len(ids)


def analyze(screen: Path, full: Path, seeds: tuple[int, ...]) -> dict:
    screen_ids, screen_scenes = ids_and_scenes(screen / "manifest.json", SCREEN_SHA)
    full_ids, full_scenes = ids_and_scenes(full / "manifest.json", FULL_SHA)
    assert len(screen_ids) == 256 and len(full_ids) == 1839
    full_scene_by_id = dict(zip(full_ids, full_scenes))
    assert set(screen_ids) <= set(full_ids)
    assert all(full_scene_by_id[eid] == scene
               for eid, scene in zip(screen_ids, screen_scenes))
    assert len(set(seeds)) == len(seeds) and set(seeds) <= {11, 22, 33}

    output = {
        "screen_manifest_sha256": SCREEN_SHA,
        "full_manifest_sha256": FULL_SHA,
        "overlap_episodes": 256,
        "models": {},
        "paired_seed_differences": {},
        "interpretation": (
            "The two evaluations use the same model labels, episode IDs, and "
            "decoding configuration, but separate stochastic inference runs. "
            "Their overlap disagreement measures realized evaluation variation; "
            "it is not a navigation improvement or an independent test set."
        ),
    }
    rows = {}
    for seed in seeds:
        for arm in ("branch", "branch_control"):
            label = f"{arm}128_seed{seed}"
            earlier = load_label(screen, label, screen_ids)
            later = load_label(full, label, full_ids)
            rows[label] = (earlier, later)
            full_only = sum(bool(later[eid]["success"]) and
                            not bool(earlier[eid]["success"]) for eid in screen_ids)
            screen_only = sum(bool(earlier[eid]["success"]) and
                              not bool(later[eid]["success"]) for eid in screen_ids)
            output["models"][label] = {
                "screen_successes": sum(bool(earlier[eid]["success"]) for eid in screen_ids),
                "full_overlap_successes": sum(bool(later[eid]["success"]) for eid in screen_ids),
                "full_only_successes": full_only,
                "screen_only_successes": screen_only,
                "changed_success_outcomes": full_only + screen_only,
                "screen_spl": metric(earlier, screen_ids, "spl"),
                "full_overlap_spl": metric(later, screen_ids, "spl"),
            }
        candidate_label = f"branch128_seed{seed}"
        control_label = f"branch_control128_seed{seed}"
        candidate_screen, candidate_full = rows[candidate_label]
        control_screen, control_full = rows[control_label]
        output["paired_seed_differences"][str(seed)] = {
            "screen_sr_pp": 100 * (metric(candidate_screen, screen_ids, "success") -
                                   metric(control_screen, screen_ids, "success")),
            "full_overlap_sr_pp": 100 * (metric(candidate_full, screen_ids, "success") -
                                         metric(control_full, screen_ids, "success")),
            "screen_spl_pp": 100 * (metric(candidate_screen, screen_ids, "spl") -
                                    metric(control_screen, screen_ids, "spl")),
            "full_overlap_spl_pp": 100 * (metric(candidate_full, screen_ids, "spl") -
                                          metric(control_full, screen_ids, "spl")),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.screen, args.full, tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
