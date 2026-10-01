"""Audit complete sharded R2R val-unseen results and matched-seed contrasts."""

import argparse
import json
import statistics
from pathlib import Path


DEFAULT_ROOT = Path("/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930/runlogs/eventtrace_full_val_unseen")
LABELS = ["sft"] + [f"seed{seed}_{arm}" for seed in (11, 22, 33)
                    for arm in ("control", "event")]


def load_arm(root, label, ids):
    rows = {}
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        expected = ids[shard::4]
        observed = [str(episode_id) for episode_id in summary["episode_ids"]]
        assert len(set(observed)) == len(observed)
        assert set(observed) == set(expected)
        assert summary["count"] == len(expected)
        assert summary["inference_errors"] == 0
        for episode_id in expected:
            record = json.loads((folder / "log" / f"stats_{episode_id}_0.json").read_text())
            assert str(record["id"]) == episode_id
            rows[episode_id] = record
    assert set(rows) == set(ids)
    return rows


def summarize(rows, ids):
    n = len(ids)
    return {
        "count": n,
        "successes": sum(bool(rows[i]["success"]) for i in ids),
        "sr": sum(bool(rows[i]["success"]) for i in ids) / n,
        "spl": sum(float(rows[i]["spl"]) for i in ids) / n,
        "mean_distance_to_goal": sum(float(rows[i]["distance_to_goal"]) for i in ids) / n,
        "inference_errors": sum(rows[i].get("early_stop_reason") == "inference_error" for i in ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    ids = manifest["episode_ids"]
    assert len(ids) == 1839 and len(set(ids)) == 1839
    rows = {label: load_arm(root, label, ids) for label in LABELS}
    result = {
        "split": "val_unseen",
        "episode_count": len(ids),
        "scene_count": len(set(manifest["scene_ids"])),
        "arms": {label: summarize(rows[label], ids) for label in LABELS},
        "paired_seed_differences": {},
    }
    for seed in (11, 22, 33):
        control, event = rows[f"seed{seed}_control"], rows[f"seed{seed}_event"]
        result["paired_seed_differences"][str(seed)] = {
            "sr": sum(float(bool(event[i]["success"])) - float(bool(control[i]["success"]))
                      for i in ids) / len(ids),
            "spl": sum(float(event[i]["spl"]) - float(control[i]["spl"])
                       for i in ids) / len(ids),
            "event_only_successes": sum(bool(event[i]["success"]) and not bool(control[i]["success"])
                                        for i in ids),
            "control_only_successes": sum(bool(control[i]["success"]) and not bool(event[i]["success"])
                                          for i in ids),
        }
    for metric in ("sr", "spl"):
        values = [result["paired_seed_differences"][str(seed)][metric]
                  for seed in (11, 22, 33)]
        result[f"mean_seed_paired_{metric}_difference"] = statistics.mean(values)
        result[f"sd_seed_paired_{metric}_difference"] = statistics.stdev(values)
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
