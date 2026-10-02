"""Compare deployed fusion formulas on reused train-scene trajectory probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


def score_subset(name: str, pairs: list[dict], cache: dict, swaps_root: Path,
                 old: TemporalPotential, new: TemporalPotential,
                 report_rows: list[dict], old_temporal_scale: float,
                 new_temporal_scale: float, visual_endpoint_scale: float,
                 visual_grounding_scale: float, blend_weight: float,
                 device: torch.device) -> dict:
    if cache["hidden"].shape != (2 * len(pairs), 4, 2048):
        raise ValueError(f"{name}: incomplete SFT cache")
    rows_by_id = {row["pair_id"]: row for row in report_rows}
    if set(rows_by_id) != {pair["pair_id"] for pair in pairs}:
        raise ValueError(f"{name}: heldout margin identity mismatch")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    wrong = []
    for index, pair in enumerate(pairs):
        for role_index, role in enumerate(("success", "failure")):
            if cache["record_ids"][2 * index + role_index] != pair["pair_id"] + "_" + role:
                raise ValueError(f"{name}: SFT identity mismatch")
        swap = torch.load(swaps_root / "records" / f"{pair['pair_id']}.pt",
                          map_location="cpu", weights_only=False)
        if swap["pair_id"] != pair["pair_id"] or swap["hidden"].shape != (4, 2048):
            raise ValueError(f"{name}: wrong-instruction identity mismatch")
        wrong.append(F.normalize(swap["hidden"].float(), dim=-1))
    wrong_hidden = torch.stack(wrong).unsqueeze(1)
    summaries = {}
    score_arrays = {}
    for model_name, model, temporal_scale in (("frozen_v2", old, old_temporal_scale),
                                               ("failure_rank_v1", new, new_temporal_scale)):
        correct = predict(model, hidden, device)
        incorrect = predict(model, wrong_hidden, device)[:, 0]
        margins = {"endpoint": [], "grounding": []}
        for index, pair in enumerate(pairs):
            visual = rows_by_id[pair["pair_id"]]["margins"]["visual"]
            # The source report normalizes grounding by a different fit
            # scale. Reconstruct raw SigLIP margins, then apply the single
            # endpoint scale used by the deployed terminal reward service.
            visual_endpoint = float(visual["endpoint"])
            visual_grounding = (float(visual["grounding"]) *
                                visual_grounding_scale / visual_endpoint_scale)
            temporal_endpoint = float(correct[index, 0, -1] -
                                      correct[index, 1, -1]) / temporal_scale
            temporal_grounding = float(correct[index, 0, -1] -
                                       incorrect[index, -1]) / temporal_scale
            margins["endpoint"].append(.5 * (temporal_endpoint + visual_endpoint))
            margins["grounding"].append(.5 * (temporal_grounding + visual_grounding))
        score_arrays[model_name] = margins
        summaries[model_name] = {task: {"hits": sum(value > 0 for value in values),
                                        "pairs": len(values),
                                        "rate": sum(value > 0 for value in values) / len(values),
                                        "mean_margin": sum(values) / len(values)}
                                 for task, values in margins.items()}
    score_arrays["residual_blend"] = {
        task: [(1 - blend_weight) * old_margin + blend_weight * new_margin
               for old_margin, new_margin in zip(score_arrays["frozen_v2"][task],
                                                 score_arrays["failure_rank_v1"][task])]
        for task in ("endpoint", "grounding")}
    summaries["residual_blend"] = {
        task: {"hits": sum(value > 0 for value in values), "pairs": len(values),
               "rate": sum(value > 0 for value in values) / len(values),
               "mean_margin": sum(values) / len(values)}
        for task, values in score_arrays["residual_blend"].items()}
    paired = {task: {"candidate_only_correct": sum(a > 0 and b <= 0 for a, b in
                                                  zip(score_arrays["failure_rank_v1"][task],
                                                      score_arrays["frozen_v2"][task])),
                     "baseline_only_correct": sum(b > 0 and a <= 0 for a, b in
                                                 zip(score_arrays["failure_rank_v1"][task],
                                                     score_arrays["frozen_v2"][task]))}
              for task in ("endpoint", "grounding")}
    return {"pairs": len(pairs), "scores": summaries, "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-pair-manifest", type=Path, required=True)
    parser.add_argument("--old-v2-manifest", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--old-swaps", type=Path, required=True)
    parser.add_argument("--novel-manifest", type=Path, required=True)
    parser.add_argument("--novel-features", type=Path, required=True)
    parser.add_argument("--novel-swaps", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--blend-development", type=Path, required=True)
    parser.add_argument("--old-encoder", type=Path, required=True)
    parser.add_argument("--new-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_pairs = json.loads(args.old_pair_manifest.read_text())["pairs"]
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    novel_pairs = json.loads(args.novel_manifest.read_text())["pairs"]
    heldout = json.loads(args.heldout_report.read_text())
    original = json.loads(args.development_report.read_text())
    calibration = json.loads(args.calibration.read_text())
    blend_development = json.loads(args.blend_development.read_text())
    if old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            calibration["new_encoder_sha256"] != digest(args.new_encoder) or \
            calibration["old_encoder_sha256"] != digest(args.old_encoder) or \
            calibration["source_development_report_sha256"] != digest(args.development_report) or \
            heldout["development_report_sha256"] != digest(args.development_report) or \
            blend_development["calibration_sha256"] != digest(args.calibration):
        raise ValueError("fusion probe provenance mismatch")
    blend_weight = blend_development["selected_weight"]
    if blend_weight is None or not 0 < blend_weight < 1:
        raise ValueError("no eligible residual blend")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    old_checkpoint = torch.load(args.old_encoder, map_location="cpu", weights_only=False)
    new_checkpoint = torch.load(args.new_encoder, map_location="cpu", weights_only=False)
    old = TemporalPotential().to(device).eval()
    old.load_state_dict(old_checkpoint["model"])
    new = TemporalPotential().to(device).eval()
    new.load_state_dict(new_checkpoint["model"])
    old_split = {row["pair_id"]: row["split"] for row in old_v2["pairs"]}
    audit_pairs = [pair for pair in old_pairs if old_split[pair["pair_id"]] == "audit"]
    if len(audit_pairs) != 48 or len(novel_pairs) != 38:
        raise ValueError("probe pair count mismatch")
    # The old feature file includes all 400 pairs; subset it in manifest
    # order using a temporary in-memory cache, without modifying source.
    old_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    positions = [index for index, pair in enumerate(old_pairs)
                 if old_split[pair["pair_id"]] == "audit"]
    audit_cache = {**old_cache,
                   "hidden": old_cache["hidden"].reshape(-1, 2, 4, 2048)[positions]
                               .reshape(-1, 4, 2048),
                   "record_ids": [old_cache["record_ids"][2 * index + role]
                                  for index in positions for role in (0, 1)]}
    v2 = score_subset("v2_audit", audit_pairs, audit_cache,
                      args.old_swaps, old, new,
                      heldout["v2_audit"]["per_pair"],
                      calibration["old_temporal_scale"],
                      calibration["temporal_scale"],
                      calibration["visual_scale"],
                      original["fit_mean_absolute_margin_scales"]["visual"]["grounding"],
                      blend_weight, device)
    novel_cache = torch.load(args.novel_features, map_location="cpu", weights_only=False)
    novel = score_subset("seed33_novel", novel_pairs, novel_cache,
                         args.novel_swaps, old, new,
                         heldout["seed33_novel"]["per_pair"],
                         calibration["old_temporal_scale"],
                         calibration["temporal_scale"],
                         calibration["visual_scale"],
                         original["fit_mean_absolute_margin_scales"]["visual"]["grounding"],
                         blend_weight, device)
    report = {"schema": "failure_rank_deployed_fusion_probe_v1",
              "interpretation": "Reused train-scene probes, including overlap with new fit scenes. The exact endpoint-calibrated terminal reward formula is evaluated; no online RL or val-unseen result.",
              "new_encoder_sha256": digest(args.new_encoder),
              "calibration_sha256": digest(args.calibration),
              "blend_development_sha256": digest(args.blend_development),
              "blend_weight": blend_weight,
              "v2_audit": v2, "seed33_novel": novel}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
