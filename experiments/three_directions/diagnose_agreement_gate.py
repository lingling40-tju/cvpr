"""Development-only precision/coverage of two frozen process signals.

This is a post-hoc diagnostic. It does not train or tune either head and
cannot substitute for a locked audit or paired navigation evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fit_group_relative_head import GroupRelativeScore
from fit_group_visual_transition import VisualTransition
from history_grounding_lora import digest, rid


INTERVALS = ((3, 6), (6, 9), (9, 12))


def counts(entries: list[dict], key: str) -> dict:
    positive = [row for row in entries if row[key] > 0]
    negative = [row for row in entries if row[key] < 0]
    actual_forward = sum(row["truth"] == "forward" for row in entries)
    actual_regression = sum(row["truth"] == "regression" for row in entries)
    positive_true = sum(row["truth"] == "forward" for row in positive)
    negative_true = sum(row["truth"] == "regression" for row in negative)
    return {"positive_decisions": len(positive),
            "positive_correct": positive_true,
            "positive_precision": positive_true / len(positive) if positive else 0,
            "forward_recall": positive_true / actual_forward,
            "negative_decisions": len(negative),
            "negative_correct": negative_true,
            "negative_precision": negative_true / len(negative) if negative else 0,
            "regression_recall": negative_true / actual_regression,
            "positive_groups": len({row["group_id"] for row in positive}),
            "negative_groups": len({row["group_id"] for row in negative}),
            "positive_episodes": len({row["episode_id"] for row in positive}),
            "negative_episodes": len({row["episode_id"] for row in negative})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--history-cache", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--history-audit", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path, required=True)
    parser.add_argument("--history-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    h_audit = json.loads(args.history_audit.read_text())
    v_audit = json.loads(args.visual_audit.read_text())
    h_ckpt = torch.load(args.history_checkpoint, map_location="cpu",
                        weights_only=True)
    v_ckpt = torch.load(args.visual_checkpoint, map_location="cpu",
                        weights_only=True)
    if (manifest["schema"] != "policy_group_relative_manifest_v1" or
            h_audit["manifest_sha256"] != manifest_sha or
            v_audit["manifest_sha256"] != manifest_sha or
            h_ckpt["manifest_sha256"] != manifest_sha or
            v_ckpt["manifest_sha256"] != manifest_sha or
            h_ckpt["source_id"] != h_audit["source_id"] or
            v_ckpt["source_id"] != v_audit["source_id"]):
        raise ValueError("audited source/checkpoint mismatch")
    history = GroupRelativeScore()
    history.load_state_dict(h_ckpt["state_dict"])
    history.eval()
    visual = VisualTransition().cuda()
    visual.load_state_dict(v_ckpt["state_dict"])
    visual.eval()
    intervals = []
    part = "development"
    for plan in manifest["selected"][part]:
        record_id = rid(plan)
        root = args.turn_root / part / "records"
        record = json.loads((root / f"{record_id}.json").read_text())
        h = torch.load(args.history_cache / part / "records" /
                       f"{record_id}.pt", map_location="cpu",
                       weights_only=True)
        v = torch.load(args.visual_cache / part / "records" /
                       f"{record_id}.pt", map_location="cpu",
                       weights_only=True)
        if (record["manifest_sha256"] != manifest_sha or
                h["manifest_sha256"] != manifest_sha or
                v["manifest_sha256"] != manifest_sha or
                h["anchor_turns"] != v["anchor_turns"] or
                h["source_id"] != h_audit["source_id"] or
                v["source_id"] != v_audit["source_id"]):
            raise ValueError(f"interval source mismatch {record_id}")
        h_state = dict(zip(h["anchor_turns"], h["hidden"]))
        v_state = dict(zip(v["anchor_turns"], v["patches"]))
        distances = {turn["original_turn_index"]:
                     turn["distance_to_goal_for_label_only"]
                     for turn in record["turns"]}
        group_id = f"s{plan['seed']}_e{plan['episode_id']}"
        for before, after in INTERVALS:
            if before not in h_state or after not in h_state:
                continue
            progress = distances[before] - distances[after]
            truth = "forward" if progress >= 1 else \
                    "regression" if progress <= -1 else "neutral"
            intervals.append({"group_id": group_id,
                              "episode_id": str(plan["episode_id"]),
                              "truth": truth,
                              "h_before": h_state[before],
                              "h_after": h_state[after],
                              "v_before": v_state[before],
                              "v_after": v_state[after],
                              "text": v["text"]})
    with torch.inference_mode():
        for start in range(0, len(intervals), 32):
            batch = intervals[start:start + 32]
            before_h = torch.stack([row["h_before"] for row in batch])
            after_h = torch.stack([row["h_after"] for row in batch])
            delta = history(after_h) - history(before_h)
            before_v = torch.stack([row["v_before"] for row in batch]).cuda()
            after_v = torch.stack([row["v_after"] for row in batch]).cuda()
            texts = torch.stack([row["text"] for row in batch]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                visual_score = visual(before_v, after_v, texts)
            for row, h_value, v_value in zip(batch, delta.tolist(),
                                              visual_score.float().tolist()):
                row["history_sign"] = 1 if h_value > 0 else -1 if h_value < 0 else 0
                row["visual_sign"] = 1 if v_value > 0 else -1 if v_value < 0 else 0
                row["agreement_sign"] = row["history_sign"] if \
                    row["history_sign"] == row["visual_sign"] else 0
    distribution = {kind: sum(row["truth"] == kind for row in intervals)
                    for kind in ("forward", "regression", "neutral")}
    scores = {kind: counts(intervals, kind + "_sign") for kind in
              ("history", "visual", "agreement")}
    agreed = scores["agreement"]
    gate = {"positive_decisions_at_least_20": agreed["positive_decisions"] >= 20,
            "negative_decisions_at_least_20": agreed["negative_decisions"] >= 20,
            "positive_precision_at_least_0_75": agreed["positive_precision"] >= .75,
            "forward_recall_at_least_0_20": agreed["forward_recall"] >= .20,
            "negative_precision_at_least_0_70": agreed["negative_precision"] >= .70,
            "regression_recall_at_least_0_20": agreed["regression_recall"] >= .20}
    report = {"schema": "agreement_process_diagnostic_v1",
              "interpretation": "Post-hoc development-only diagnostic; no locked audit or RL.",
              "manifest_sha256": manifest_sha,
              "history_checkpoint_sha256": digest(args.history_checkpoint),
              "visual_checkpoint_sha256": digest(args.visual_checkpoint),
              "intervals": len(intervals),
              "unique_groups": len({row["group_id"] for row in intervals}),
              "unique_episodes": len({row["episode_id"] for row in intervals}),
              "truth_distribution": distribution,
              "scores": scores, "diagnostic_gate": gate,
              "passed": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
