"""Describe stop-aware termination behavior on matched held-out episodes.

This post-hoc analysis does not identify a causal mechanism. It reads the
frozen 256-episode screen only after both model labels pass full validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


MANIFEST_SHA256 = "2f8d1438921f2030c5036a9af5b823cf956be57ce0496be42dbafa9b595d6375"


def load_label(root: Path, label: str, ids: list[str]) -> dict[str, dict]:
    if not (root / f"{label}.completed").is_file():
        raise ValueError(f"incomplete label {label}")
    rows = {}
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        expected = ids[shard::4]
        observed = list(map(str, summary["episode_ids"]))
        if summary["count"] != len(expected) or set(observed) != set(expected) or \
                len(observed) != len(set(observed)) or summary["inference_errors"]:
            raise ValueError(f"invalid shard {label}/{shard}")
        for eid in expected:
            row = json.loads((folder / "log" / f"stats_{eid}_0.json").read_text())
            if str(row["id"]) != eid or row.get("early_stop_reason") == "inference_error":
                raise ValueError(f"invalid episode {label}/{eid}")
            rows[eid] = row
    if set(rows) != set(ids):
        raise ValueError(f"episode coverage mismatch {label}")
    return rows


def summary(rows: dict[str, dict], ids: list[str]) -> dict:
    return {"episodes": len(ids),
            "successes": sum(bool(rows[i]["success"]) for i in ids),
            "oracle_successes": sum(bool(rows[i]["oracle_success"]) for i in ids),
            "early_stop_reasons": dict(Counter(str(rows[i].get("early_stop_reason"))
                                               for i in ids)),
            "mean_path_length_m": statistics.mean(float(rows[i]["path_length"])
                                                    for i in ids),
            "mean_distance_to_goal_m": statistics.mean(
                float(rows[i]["distance_to_goal"]) for i in ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--train-rollout", type=Path, required=True)
    parser.add_argument("--candidate", default="stopaware_group4_64_seed11")
    parser.add_argument("--control", default="group4_64_seed11_fresh256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_manifest = (args.eval_root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("pilot manifest mismatch")
    manifest = json.loads(raw_manifest)
    ids = list(map(str, manifest["episode_ids"]))
    scenes = list(map(str, manifest["scene_ids"]))
    if len(ids) != 256 or len(set(ids)) != 256 or len(scenes) != 256 or \
            len(set(scenes)) != 10:
        raise ValueError("unexpected disjoint evaluation manifest")
    candidate = load_label(args.eval_root, args.candidate, ids)
    control = load_label(args.eval_root, args.control, ids)
    by_outcome = {
        "candidate_only": [i for i in ids if candidate[i]["success"] and not control[i]["success"]],
        "control_only": [i for i in ids if control[i]["success"] and not candidate[i]["success"]],
        "both_failure": [i for i in ids if not candidate[i]["success"] and not control[i]["success"]],
        "both_success": [i for i in ids if candidate[i]["success"] and control[i]["success"]],
    }
    groups = {name: {"episodes": len(selected),
                     "candidate_reasons": dict(Counter(
                         str(candidate[i].get("early_stop_reason")) for i in selected)),
                     "control_reasons": dict(Counter(
                         str(control[i].get("early_stop_reason")) for i in selected))}
              for name, selected in by_outcome.items()}
    train = [json.loads(line) for line in args.train_rollout.read_text().splitlines()]
    if len(train) != 64 or [row["step"] for row in train] != list(range(1, 65)):
        raise ValueError("incomplete training rollout")
    trajectories = [item for row in train for item in row["info"]]
    if len(trajectories) != 1024:
        raise ValueError("unexpected training rollout count")
    failure = [item for item in trajectories if not item["task_success"]]
    terminal = {}
    for reason in sorted({str(item["end_reason"]) for item in failure}):
        selected = [item for item in failure if str(item["end_reason"]) == reason]
        terminal[reason] = {
            "rollouts": len(selected),
            "positive_bonus_rollouts": sum(
                float(item["reward_components"]["fused_bonus"]) > 0
                for item in selected),
            "mean_applied_bonus": statistics.mean(
                float(item["reward_components"]["fused_bonus"])
                for item in selected),
            "terminal_distance_below_3_5_m": sum(
                float(item["distance_to_goal"]) < 3.5 for item in selected)}
    result = {"schema": "stopaware_matched_val256_modes_v1",
              "interpretation": "Post-hoc behavior description on a disjoint 256-episode val-unseen screen and candidate train rollouts. It cannot identify causality or prove another reward will help.",
              "manifest_sha256": MANIFEST_SHA256,
              "candidate": args.candidate, "control": args.control,
              "candidate_val": summary(candidate, ids),
              "control_val": summary(control, ids),
              "paired_outcome_groups": groups,
              "candidate_train_failure_terminal_modes": terminal}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
