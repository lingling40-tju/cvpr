"""One locked train-scene audit of the selected two-head readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stop_history_head import StopProgressHead, digest, load_part, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("locked audit output already exists; do not peek repeatedly")
    manifest_sha = digest(args.manifest)
    labels = json.loads(args.label_audit.read_text())
    fit_report = json.loads((args.fit_root / "fit_report.json").read_text())
    checkpoint = torch.load(args.fit_root / "head.pt", map_location="cpu",
                            weights_only=True)
    summary = json.loads((args.features_root / "audit" / "summary.json").read_text())
    if (labels["manifest_sha256"] != manifest_sha or
            fit_report["manifest_sha256"] != manifest_sha or
            checkpoint["manifest_sha256"] != manifest_sha or
            checkpoint["label_audit_sha256"] != digest(args.label_audit) or
            checkpoint["schema"] != "stop_history_two_head_seed11_v1" or
            checkpoint["seed"] != 11 or fit_report["seed"] != 11 or
            checkpoint["selected_epoch"] != fit_report["selected_epoch"] or
            checkpoint["development_stop_threshold"] !=
            fit_report["development_stop_threshold"]["threshold"] or
            summary["manifest_sha256"] != manifest_sha or
            summary["requested"] != summary["completed"] or
            summary["completed"] != 112 or summary["errors"] or
            summary["limit"] != 0):
        raise ValueError("source/checkpoint/feature coverage mismatch")
    audit = load_part(args.features_root, labels, "audit", manifest_sha)
    if len(audit["episode_ids"]) != 112:
        raise ValueError("audit trajectory coverage mismatch")
    state = checkpoint["model_state"]
    model = StopProgressHead(state["fit_mean"], state["fit_std"])
    model.load_state_dict(state)
    scored = metrics(model, audit)
    threshold = float(checkpoint["development_stop_threshold"])
    recall = sum(value >= threshold for value in scored["positive_logits"]) / \
        len(scored["positive_logits"])
    fpr = sum(value >= threshold for value in scored["negative_logits"]) / \
        len(scored["negative_logits"])
    scored.pop("positive_logits")
    scored.pop("negative_logits")
    gate = {
        "min_100_local_progress_pairs": scored["progress_pairs"] >= 100,
        "min_100_safe_instruction_swaps": scored["instruction_swap_pairs"] >= 100,
        "progress_pair_accuracy_at_least_0_75": scored["progress_pair_accuracy"] >= 0.75,
        "instruction_swap_accuracy_at_least_0_75": scored["instruction_swap_accuracy"] >= 0.75,
        "stop_auc_at_least_0_80": scored["stop_auc"] >= 0.80,
        "stop_false_positive_rate_at_most_0_10": fpr <= 0.10,
        "stop_recall_at_least_0_50": recall >= 0.50,
    }
    result = {
        "schema": "stop_history_two_head_locked_audit_v1",
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": digest(args.fit_root / "head.pt"),
        "fit_report_sha256": digest(args.fit_root / "fit_report.json"),
        "selection": "seed 11, lowest development composite loss and development-selected STOP threshold; no audit-based checkpoint selection",
        "audit_scenes": len(set(audit["scenes"])),
        "audit_trajectories": len(audit["episode_ids"]),
        "metrics": scored,
        "development_selected_stop_threshold": threshold,
        "stop_false_positive_rate_at_threshold": fpr,
        "stop_recall_at_threshold": recall,
        "predeclared_gates": gate,
        "eligible_for_group4_rl_pilot": all(gate.values()),
        "interpretation": "Train-scene held-out screen only; not val-unseen navigation evidence. The underlying SFT model has seen R2R train data."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
