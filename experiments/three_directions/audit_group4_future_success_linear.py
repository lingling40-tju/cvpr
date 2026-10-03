"""One-time model-held-out audit of frozen group-four outcome value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest, load_part, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--collection-audit", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    development = json.loads(args.development_report.read_text())
    collection = json.loads(args.collection_audit.read_text())
    state_audit = json.loads(args.state_audit.read_text())
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    manifest_sha = digest(args.manifest)
    if manifest["schema"] != "policy_group_relative_manifest_v1" or \
            development["schema"] != "group4_future_success_linear_development_v1" or \
            not development["eligible_for_one_time_model_audit"] or \
            not all(development["predeclared_gate"].values()) or \
            development["manifest_sha256"] != manifest_sha or \
            weights["schema"] != "group4_future_success_linear_weights_v1" or \
            weights["manifest_sha256"] != manifest_sha or \
            weights["encoder_source_id"] != development["encoder_source_id"] or \
            state_audit["manifest_sha256"] != manifest_sha or \
            state_audit["source_id"] != development["encoder_source_id"] or \
            collection["manifest_sha256"] != manifest_sha:
        raise ValueError("frozen model or source provenance mismatch")
    coverage, pairs = load_part(manifest, "audit", args.turn_root,
                                args.state_root, manifest_sha,
                                state_audit["source_id"])
    if coverage["trajectories"] != state_audit["parts"]["audit"]["trajectories"] or \
            coverage["states"] > state_audit["parts"]["audit"]["states"]:
        raise ValueError("incomplete audit cache")
    scale = weights["scale"].float()
    vector = weights["vector"].float()
    if scale.shape != (2048,) or vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all() and torch.isfinite(vector).all()):
        raise ValueError("invalid frozen model weights")
    data = F.normalize(torch.stack([pair["difference"] for pair in pairs]) /
                       scale, dim=1)
    scores = data @ vector
    outcome = metrics(pairs, scores)
    gate = {"audit_pairs_at_least_100": coverage["matched_pairs"] >= 100,
            "audit_groups_at_least_20": coverage["matched_groups"] >= 20,
            "comparison_accuracy_at_least_0_70": outcome["accuracy"] >= .70,
            "group_macro_accuracy_at_least_0_70":
                outcome["group_macro_accuracy"] >= .70}
    result = {"schema": "group4_future_success_linear_locked_audit_v1",
              "interpretation": "one-time model-held-out train-scene audit, with research-wide scene reuse; not navigation gain",
              "manifest_sha256": manifest_sha,
              "development_report_sha256": digest(args.development_report),
              "weights_sha256": digest(args.weights),
              "collection_audit_sha256": digest(args.collection_audit),
              "state_audit_sha256": digest(args.state_audit),
              "coverage": coverage,
              "metrics": outcome,
              "predeclared_gate": gate,
              "eligible_for_instruction_grounding_and_group4_wiring": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
