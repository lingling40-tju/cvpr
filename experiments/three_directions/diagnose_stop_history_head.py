"""Post-hoc failure breakdown for the locked STOP/progress audit.

This script runs only after the locked audit has been written. Its output
is diagnostic and must not be used to retune the same checkpoint or gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stop_history_head import StopProgressHead, digest, load_part


def rate(scores: torch.Tensor, threshold: float) -> dict:
    return {"count": len(scores),
            "at_or_above_development_threshold": int((scores >= threshold).sum()),
            "rate": float((scores >= threshold).float().mean()) if len(scores) else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locked = args.fit_root / "locked_audit.json"
    if not locked.is_file():
        raise ValueError("locked audit must precede post-hoc diagnosis")
    manifest_sha = digest(args.manifest)
    labels = json.loads(args.label_audit.read_text())
    checkpoint = torch.load(args.fit_root / "head.pt", map_location="cpu",
                            weights_only=True)
    audit = json.loads(locked.read_text())
    if (labels["manifest_sha256"] != manifest_sha or
            checkpoint["manifest_sha256"] != manifest_sha or
            audit["checkpoint_sha256"] != digest(args.fit_root / "head.pt")):
        raise ValueError("source/locked audit mismatch")
    model = StopProgressHead(checkpoint["model_state"]["fit_mean"],
                             checkpoint["model_state"]["fit_std"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint["development_stop_threshold"])
    result = {"schema": "stop_history_posthoc_mode_breakdown_v1",
              "locked_audit_sha256": digest(locked),
              "interpretation": "Post-hoc diagnosis only. The 10% STOP false-positive gate remains failed; do not select a new threshold using these audit rows.",
              "parts": {}}
    for part in ("development", "audit"):
        data = load_part(args.features_root, labels, part, manifest_sha)
        with torch.inference_mode():
            stop, progress = model(data["hidden"])
            wrong, _ = model(data["wrong_hidden"])
        distance = data["distances"]
        safe = data["safe_wrong_mask"]
        gap = distance[:, :-1] - distance[:, 1:]
        valid = gap.abs() >= 1.0
        result["parts"][part] = {
            "correct_terminal_positive": rate(stop[:, 2], threshold),
            "correct_initial_far_negative": rate(stop[:, 0], threshold),
            "correct_mid_far_negative": rate(stop[:, 1][distance[:, 1] > 3.0], threshold),
            "correct_mid_near_positive": rate(stop[:, 1][distance[:, 1] <= 3.0], threshold),
            "verified_wrong_instruction_terminal_negative": rate(wrong[safe], threshold),
            "progress_comparison_pairs": int(valid.sum()),
            "progress_decreasing_distance_pairs": int(((gap >= 1.0) & valid).sum()),
            "progress_increasing_distance_pairs": int(((gap <= -1.0) & valid).sum()),
            "progress_score_correct_on_increasing_distance_pairs": int((
                ((progress[:, 1:] - progress[:, :-1]) < 0) &
                (gap <= -1.0)).sum()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
