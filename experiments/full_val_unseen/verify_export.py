"""Recompute archived full val-unseen summaries from the compact CSV."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABELS = ["sft"] + [f"seed{seed}_{arm}" for seed in (11, 22, 33)
                    for arm in ("control", "event")]


def close(actual, expected):
    assert abs(actual - expected) < 1e-10, (actual, expected)


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    analysis = json.loads((ROOT / "analysis.json").read_text())
    ids = list(map(str, manifest["episode_ids"]))
    scenes = list(map(str, manifest["scene_ids"]))
    assert len(ids) == len(scenes) == 1839 and len(set(ids)) == 1839
    assert len(set(scenes)) == analysis["scene_count"] == 11
    rows = {label: {} for label in LABELS}
    with (ROOT / "episodes.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            label, episode_id = row["label"], row["episode_id"]
            assert label in rows and episode_id not in rows[label]
            rows[label][episode_id] = row
    for label in LABELS:
        arm = rows[label]
        assert list(arm) == ids
        assert all(arm[i]["scene_id"] == scene for i, scene in zip(ids, scenes))
        assert all(arm[i]["early_stop_reason"] != "inference_error" for i in ids)
        expected = analysis["arms"][label]
        successes = sum(float(arm[i]["success"]) != 0 for i in ids)
        assert successes == expected["successes"]
        close(successes / len(ids), expected["sr"])
        close(statistics.mean(float(arm[i]["spl"]) for i in ids), expected["spl"])
        close(statistics.mean(float(arm[i]["distance_to_goal"]) for i in ids),
              expected["mean_distance_to_goal"])
        validated = json.loads((ROOT / f"{label}.validated.json").read_text())
        assert validated == {"label": label, "episodes": len(ids),
                             "successes": successes, "inference_errors": 0}
    for seed in (11, 22, 33):
        control = rows[f"seed{seed}_control"]
        event = rows[f"seed{seed}_event"]
        expected = analysis["paired_seed_differences"][str(seed)]
        close(statistics.mean(float(event[i]["success"]) -
                              float(control[i]["success"]) for i in ids), expected["sr"])
        close(statistics.mean(float(event[i]["spl"]) -
                              float(control[i]["spl"]) for i in ids), expected["spl"])
        assert sum(float(event[i]["success"]) != 0 and
                   float(control[i]["success"]) == 0 for i in ids) == expected["event_only_successes"]
        assert sum(float(control[i]["success"]) != 0 and
                   float(event[i]["success"]) == 0 for i in ids) == expected["control_only_successes"]
    print("Verified seven models, 12,873 unique rows, matched 1,839-episode coverage, "
          "zero inference errors, summaries, and paired seed contrasts.")


if __name__ == "__main__":
    main()
