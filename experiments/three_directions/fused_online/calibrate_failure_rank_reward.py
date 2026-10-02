"""Fit-only scale for the failure-aware temporal reward representation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


DEVELOPMENT_REPORT_SHA256 = "d5ce03fd71efa7b264cc3c064c39f40df1bcbae1ec9f6fd5d7e43664ccc434c4"


def scores(model: TemporalPotential, hidden: torch.Tensor,
           indices: list[int], device: torch.device) -> torch.Tensor:
    selected = torch.tensor(indices, dtype=torch.long)
    output = predict(model, hidden[selected], device)
    return output[:, 0, -1] - output[:, 1, -1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--failure-features", type=Path, required=True)
    parser.add_argument("--new-encoder", type=Path, required=True)
    parser.add_argument("--old-pair-manifest", type=Path, required=True)
    parser.add_argument("--old-v2-manifest", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--old-encoder", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failure = json.loads(args.failure_manifest.read_text())
    old_pairs = json.loads(args.old_pair_manifest.read_text())["pairs"]
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    fusion = json.loads(args.fusion_report.read_text())
    if digest(args.fusion_report) != DEVELOPMENT_REPORT_SHA256 or \
            fusion["schema"] != "equal_temporal_visual_reward_development_v3_online_parity" or \
            old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest):
        raise ValueError("frozen reference provenance mismatch")
    new_checkpoint = torch.load(args.new_encoder, map_location="cpu", weights_only=False)
    old_checkpoint = torch.load(args.old_encoder, map_location="cpu", weights_only=False)
    if new_checkpoint["manifest_sha256"] != digest(args.failure_manifest) or \
            new_checkpoint["old_encoder_sha256"] != digest(args.old_encoder):
        raise ValueError("new encoder provenance mismatch")
    new_cache = torch.load(args.failure_features, map_location="cpu", weights_only=False)
    old_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    if new_cache["manifest_sha256"] != digest(args.failure_manifest) or \
            new_cache["hidden"].shape != (2 * len(failure["pairs"]), 4, 2048) or \
            old_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            old_cache["hidden"].shape != (2 * len(old_pairs), 4, 2048):
        raise ValueError("feature caches incomplete")
    new_hidden = F.normalize(new_cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    old_hidden = F.normalize(old_cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    failure_fit = [index for index, pair in enumerate(failure["pairs"])
                   if pair["split"] == "fit"]
    old_split = {row["pair_id"]: row["split"] for row in old_v2["pairs"]}
    success_fit = [index for index, pair in enumerate(old_pairs)
                   if old_split[pair["pair_id"]] == "fit"]
    if len(failure_fit) != 697 or len(success_fit) != 300:
        raise ValueError("fit split count mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate = TemporalPotential().to(device).eval()
    candidate.load_state_dict(new_checkpoint["model"])
    baseline = TemporalPotential().to(device).eval()
    baseline.load_state_dict(old_checkpoint["model"])
    failure_margin = scores(candidate, new_hidden, failure_fit, device)
    success_margin = scores(candidate, old_hidden, success_fit, device)
    baseline_success_margin = scores(baseline, old_hidden, success_fit, device)
    failure_scale = float(failure_margin.abs().mean())
    success_scale = float(success_margin.abs().mean())
    temporal_scale = math.sqrt(failure_scale * success_scale)
    visual_scale = float(fusion["fit_mean_absolute_margin_scales"]
                         ["visual"]["endpoint"])
    old_temporal_scale = float(fusion["fit_mean_absolute_margin_scales"]
                               ["temporal"]["endpoint"])
    if min(failure_scale, success_scale, temporal_scale, visual_scale) <= .01 or \
            not all(map(math.isfinite, (failure_scale, success_scale,
                                        temporal_scale, visual_scale))):
        raise ValueError("invalid fit-only reward scale")
    report = {"schema": "failure_rank_reward_calibration_v1",
              "interpretation": "Fit scenes only; no development/audit selection or online navigation result.",
              "failure_rank_manifest_sha256": digest(args.failure_manifest),
              "new_encoder_sha256": digest(args.new_encoder),
              "old_encoder_sha256": digest(args.old_encoder),
              "source_development_report_sha256": DEVELOPMENT_REPORT_SHA256,
              "fit_counts": {"failure_rank": len(failure_fit),
                             "success_failure": len(success_fit)},
              "fit_mean_absolute_margins": {
                  "new_failure_rank": failure_scale,
                  "new_success_failure": success_scale,
                  "old_success_failure": float(baseline_success_margin.abs().mean())},
              "temporal_scale": temporal_scale,
              "visual_scale": visual_scale,
              "old_temporal_scale": old_temporal_scale,
              "formula": "sigmoid(0.5*(new_temporal_score/temporal_scale + fixed_siglip_score/visual_scale))"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
