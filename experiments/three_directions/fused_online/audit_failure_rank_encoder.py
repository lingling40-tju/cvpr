"""Open the failure-rank scene audit after the development gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict
from train_failure_rank_encoder import evaluate, MODEL_CONFIG_SHA256, OLD_ENCODER_SHA256


def retention(model: TemporalPotential, hidden: torch.Tensor,
              wrong: torch.Tensor, device: torch.device) -> tuple[list[bool], list[bool]]:
    correct = predict(model, hidden, device)
    swapped = predict(model, wrong.unsqueeze(1), device)[:, 0]
    endpoint = [bool(value > 0) for value in
                correct[:, 0, -1] - correct[:, 1, -1]]
    grounding = [bool(value > 0) for value in
                 correct[:, 0, -1] - swapped[:, -1]]
    return endpoint, grounding


def paired_summary(candidate: list[bool], baseline: list[bool]) -> dict:
    if len(candidate) != len(baseline):
        raise ValueError("retention pair count mismatch")
    return {"pairs": len(candidate),
            "candidate_hits": sum(candidate), "baseline_hits": sum(baseline),
            "candidate_rate": sum(candidate) / len(candidate),
            "baseline_rate": sum(baseline) / len(baseline),
            "candidate_only_correct": sum(a and not b for a, b in zip(candidate, baseline)),
            "baseline_only_correct": sum(b and not a for a, b in zip(candidate, baseline))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--old-encoder", type=Path, required=True)
    parser.add_argument("--old-pair-manifest", type=Path, required=True)
    parser.add_argument("--old-v2-manifest", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--old-swaps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    prior = json.loads(args.development.read_text())
    candidate_checkpoint = torch.load(args.checkpoint, map_location="cpu",
                                      weights_only=False)
    old_checkpoint = torch.load(args.old_encoder, map_location="cpu",
                                weights_only=False)
    manifest_hash = digest(args.manifest)
    if manifest["schema"] != "failure_rank_train_scene_v1" or \
            not all(prior["predeclared_gate"].values()) or \
            not (prior["manifest_sha256"] == candidate_checkpoint["manifest_sha256"] == manifest_hash) or \
            digest(args.old_encoder) != OLD_ENCODER_SHA256 or \
            candidate_checkpoint["old_encoder_sha256"] != OLD_ENCODER_SHA256:
        raise ValueError("development gate or encoder provenance invalid")
    pairs = manifest["pairs"]
    indices = [index for index, pair in enumerate(pairs) if pair["split"] == "audit"]
    if len(indices) != manifest["counts"]["audit"]:
        raise ValueError("failure-rank audit split mismatch")
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if cache["manifest_sha256"] != manifest_hash or \
            cache["model_config_sha256"] != MODEL_CONFIG_SHA256 or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048):
        raise ValueError("failure-rank feature cache mismatch")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    stop = cache["stop_margin"].float().reshape(-1, 2, 4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = TemporalPotential().to(device).eval()
    baseline.load_state_dict(old_checkpoint["model"])
    candidate = TemporalPotential().to(device).eval()
    candidate.load_state_dict(candidate_checkpoint["model"])
    old_rank = evaluate(baseline, hidden, stop, indices, pairs, device, bootstrap=True)
    new_rank = evaluate(candidate, hidden, stop, indices, pairs, device, bootstrap=True)

    old_pairs = json.loads(args.old_pair_manifest.read_text())["pairs"]
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    if old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest):
        raise ValueError("old pair/v2 manifest mismatch")
    split_by_id = {row["pair_id"]: row["split"] for row in old_v2["pairs"]}
    old_indices = [index for index, pair in enumerate(old_pairs)
                   if split_by_id[pair["pair_id"]] == "audit"]
    if len(old_indices) != 48:
        raise ValueError("old success/grounding audit count mismatch")
    old_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    if old_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            old_cache["model_config_sha256"] != MODEL_CONFIG_SHA256 or \
            old_cache["hidden"].shape != (2 * len(old_pairs), 4, 2048):
        raise ValueError("old feature cache mismatch")
    old_hidden = F.normalize(old_cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    selected = old_hidden[old_indices]
    wrong = []
    for index in old_indices:
        pair = old_pairs[index]
        for role_index, role in enumerate(("success", "failure")):
            if old_cache["record_ids"][2 * index + role_index] != pair["pair_id"] + "_" + role:
                raise ValueError("old feature identity mismatch")
        swap = torch.load(args.old_swaps / "records" /
                          f"{pair['pair_id']}.pt", map_location="cpu",
                          weights_only=False)
        if swap["pair_id"] != pair["pair_id"] or \
                swap["manifest_sha256"] != digest(args.old_v2_manifest) or \
                swap["model_config_sha256"] != MODEL_CONFIG_SHA256 or \
                swap["hidden"].shape != (4, 2048):
            raise ValueError("old swapped-instruction feature mismatch")
        wrong.append(F.normalize(swap["hidden"].float(), dim=-1))
    wrong_hidden = torch.stack(wrong)
    old_endpoint, old_grounding = retention(baseline, selected, wrong_hidden, device)
    new_endpoint, new_grounding = retention(candidate, selected, wrong_hidden, device)
    endpoint = paired_summary(new_endpoint, old_endpoint)
    grounding = paired_summary(new_grounding, old_grounding)
    rank_gain = (new_rank["nearer_failure_above_farther_rate"] -
                 old_rank["nearer_failure_above_farther_rate"])
    gate = {"audit_failure_rank_at_least_0_65":
                new_rank["nearer_failure_above_farther_rate"] >= .65,
            "audit_gain_over_frozen_v2_at_least_5pp": rank_gain >= .05,
            "reused_success_endpoint_drop_at_most_5pp":
                endpoint["candidate_rate"] >= endpoint["baseline_rate"] - .05,
            "reused_instruction_grounding_drop_at_most_5pp":
                grounding["candidate_rate"] >= grounding["baseline_rate"] - .05}
    report = {"schema": "failure_rank_encoder_scene_audit_v1",
              "interpretation": "Train-scene audit, plus reused older success/grounding audit. No online RL or val-unseen navigation result.",
              "manifest_sha256": manifest_hash,
              "development_report_sha256": digest(args.development),
              "checkpoint_sha256": digest(args.checkpoint),
              "old_encoder_sha256": OLD_ENCODER_SHA256,
              "failure_rank_baseline": old_rank,
              "failure_rank_candidate": new_rank,
              "reused_success_endpoint": endpoint,
              "reused_instruction_grounding": grounding,
              "predeclared_gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
