"""Read the fresh scene audit once for a frozen instruction-aware encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, evaluate, predict
from train_temporal_contrastive_v2 import instruction_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--swaps-root", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    prior = json.loads(args.development.read_text())
    frozen = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    v2_hash = digest(args.v2_manifest)
    if v2["source_manifest_sha256"] != digest(args.manifest) or \
            prior["v2_manifest_sha256"] != v2_hash or \
            frozen["v2_manifest_sha256"] != v2_hash or \
            not all(prior["predeclared_gate"].values()):
        raise ValueError("development gate or provenance invalid")
    pairs = source["pairs"]
    rows = {row["pair_id"]: row for row in v2["pairs"]}
    indices = [i for i, pair in enumerate(pairs)
               if rows[pair["pair_id"]]["split"] == "audit"]
    if len(indices) != v2["counts"]["audit"]:
        raise ValueError("audit split count mismatch")
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    label_summary = json.loads((args.labels_root / "summary.json").read_text())
    swap_summary = json.loads((args.swaps_root / "summary_audit.json").read_text())
    if cache["manifest_sha256"] != digest(args.manifest) or \
            cache["model_config_sha256"] != frozen["feature_model_config_sha256"] or \
            label_summary["completed_trajectories"] != 2 * len(pairs) or \
            label_summary["errors"] or \
            swap_summary["v2_manifest_sha256"] != v2_hash or \
            swap_summary["records"] != len(indices):
        raise ValueError("incomplete audit cache")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    stop = cache["stop_margin"].float().reshape(-1, 2, 4)
    truth = torch.zeros(len(pairs), 2, 4)
    wrong = torch.zeros(len(pairs), 4, 2048)
    for index in indices:
        pair = pairs[index]
        for role_index, role in enumerate(("success", "failure")):
            record_id = pair["pair_id"] + "_" + role
            if cache["record_ids"][2 * index + role_index] != record_id:
                raise ValueError(f"feature identity mismatch {record_id}")
            item = json.loads((args.labels_root / "records" /
                               f"{record_id}.json").read_text())
            if item["record_id"] != record_id:
                raise ValueError(f"geodesic identity mismatch {record_id}")
            distances = item["distance_to_goal_m"]
            if len(distances) != 4 or distances[0] <= 0:
                raise ValueError(f"invalid geodesic label {record_id}")
            truth[index, role_index] = torch.tensor(
                [(distances[0] - distance) / distances[0]
                 for distance in distances])
        swap = torch.load(args.swaps_root / "records" /
                          f"{pair['pair_id']}.pt", map_location="cpu",
                          weights_only=False)
        if swap["pair_id"] != pair["pair_id"] or \
                swap["manifest_sha256"] != v2_hash or \
                swap["model_config_sha256"] != cache["model_config_sha256"] or \
                swap["hidden"].shape != (4, 2048):
            raise ValueError(f"wrong-instruction feature mismatch {pair['pair_id']}")
        wrong[index] = F.normalize(swap["hidden"].float(), dim=-1)
    model = TemporalPotential().cuda().eval()
    model.load_state_dict(frozen["model"])
    correct_pred = predict(model, hidden, torch.device("cuda:0"))
    wrong_pred = predict(model, wrong.unsqueeze(1), torch.device("cuda:0"))[:, 0]
    endpoint = evaluate(correct_pred, truth, indices, pairs, stop)
    grounding = instruction_score(correct_pred, wrong_pred, indices, pairs)
    gate = {"audit_endpoint_at_least_0_75":
                endpoint["endpoint_success_over_failure"] >= .75,
            "audit_temporal_at_least_0_60":
                endpoint["temporal_concordance"] >= .60,
            "audit_instruction_at_least_0_75":
                grounding["correct_instruction_preference"] >= .75,
            "audit_endpoint_gain_over_raw_stop_at_least_5pp":
                endpoint["endpoint_success_over_failure"] -
                endpoint["raw_stop_reference"] >= .05,
            "positive_instruction_margin": grounding["mean_correct_minus_wrong"] > 0}
    report = {"schema": "temporal_contrastive_v2_audit_v1",
              "interpretation": "Fresh train-scene audit; no online RL or val-unseen result.",
              "source_manifest_sha256": digest(args.manifest),
              "v2_manifest_sha256": v2_hash,
              "development_report_sha256": digest(args.development),
              "checkpoint_sha256": digest(args.checkpoint),
              "endpoint_and_temporal": endpoint,
              "instruction_grounding": grounding,
              "predeclared_gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
