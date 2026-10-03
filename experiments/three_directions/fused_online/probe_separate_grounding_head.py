"""Fit a separate instruction-grounding readout on frozen navigation states.

The older temporal encoder's scalar mixes route progress and instruction
grounding. This screen freezes its 64-dimensional penultimate representation
and learns only a convex linear readout from correct-vs-swapped instruction
pairs in fit scenes. Development is opened once; the audit is opened only
if fixed development gates pass. This cannot establish navigation benefit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def representation(model: TemporalPotential, hidden: torch.Tensor) -> torch.Tensor:
    """Return frozen terminal-minus-initial penultimate feature [batch, 64]."""
    rows = []
    with torch.inference_mode():
        for item in hidden.split(64):
            reference = item[:, :1]
            state = model.project(torch.cat((item, item - reference), dim=-1))
            mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
            state = model.temporal(state + model.position, mask=mask)
            neck = model.head[2](model.head[1](model.head[0](state)))
            rows.append(neck[:, -1] - neck[:, 0])
    return torch.cat(rows)


def summary(correct: torch.Tensor, wrong: torch.Tensor, weight: torch.Tensor,
            threshold: float) -> dict:
    c = correct @ weight
    w = wrong @ weight
    return {"pairs": len(c), "correct_over_wrong_hits": int((c > w).sum()),
            "correct_over_wrong_rate": float((c > w).float().mean()),
            "correct_acceptance": float((c > threshold).float().mean()),
            "wrong_false_positive": float((w > threshold).float().mean()),
            "mean_pair_margin": float((c - w).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("pair_manifest", "split_manifest", "features", "swaps",
                 "encoder", "output", "checkpoint"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    pairs = json.loads(args.pair_manifest.read_text())["pairs"]
    split_manifest = json.loads(args.split_manifest.read_text())
    if split_manifest["source_manifest_sha256"] != digest(args.pair_manifest) or \
            len(pairs) != 400:
        raise ValueError("pair/split provenance mismatch")
    assigned = {row["pair_id"]: row["split"] for row in split_manifest["pairs"]}
    splits = {name: [i for i, pair in enumerate(pairs)
                     if assigned[pair["pair_id"]] == name]
              for name in ("fit", "development", "audit")}
    if [len(splits[name]) for name in splits] != [300, 52, 48]:
        raise ValueError("unexpected scene split sizes")
    if any({pairs[i]["scene_id"] for i in splits[a]} &
           {pairs[j]["scene_id"] for j in splits[b]}
           for a, b in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit"))):
        raise ValueError("scene split leakage")
    cache = torch.load(args.features, map_location="cpu", weights_only=True)
    if cache["manifest_sha256"] != digest(args.pair_manifest) or \
            tuple(cache["hidden"].shape) != (800, 4, 2048):
        raise ValueError("navigation state cache mismatch")
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=True)
    if checkpoint["source_manifest_sha256"] != digest(args.pair_manifest) or \
            checkpoint["v2_manifest_sha256"] != digest(args.split_manifest):
        raise ValueError("encoder provenance mismatch")
    model = TemporalPotential().eval()
    model.load_state_dict(checkpoint["model"])
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(400, 2, 4, 2048)
    correct = representation(model, hidden[:, 0])
    old_weight = model.head[3].weight.detach().reshape(-1).clone()
    with torch.inference_mode():
        old_score = model(hidden[:, 0])[:, -1]
    if (correct @ old_weight - old_score).abs().max() > 1e-4:
        raise ValueError("penultimate feature does not reproduce frozen score")

    def wrong_features(indices: list[int]) -> torch.Tensor:
        states = []
        for index in indices:
            pair_id = pairs[index]["pair_id"]
            row = torch.load(args.swaps / "records" / f"{pair_id}.pt",
                             map_location="cpu", weights_only=True)
            if row["pair_id"] != pair_id or tuple(row["hidden"].shape) != (4, 2048):
                raise ValueError("wrong-instruction cache mismatch")
            states.append(row["hidden"].float())
        return representation(model, F.normalize(torch.stack(states), dim=-1))

    active = splits["fit"] + splits["development"]
    wrong_active = wrong_features(active)
    wrong_by_index = {index: value for index, value in zip(active, wrong_active)}
    fit = torch.tensor(splits["fit"], dtype=torch.long)
    dev = torch.tensor(splits["development"], dtype=torch.long)
    fit_wrong = torch.stack([wrong_by_index[int(i)] for i in fit])
    dev_wrong = torch.stack([wrong_by_index[int(i)] for i in dev])
    difference = correct[fit] - fit_wrong
    weight = torch.nn.Parameter(old_weight.clone())
    optimizer = torch.optim.LBFGS([weight], lr=.5, max_iter=200,
                                  tolerance_grad=1e-8, tolerance_change=1e-10)
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        margin = difference @ weight
        loss = F.softplus(.3 - margin).mean() + \
            .1 * (weight - old_weight).square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("nonfinite grounding head")
    # Fit-only wrong-instruction scores set a 10% false-positive target.
    wrong_scores = sorted((fit_wrong @ weight.detach()).tolist())
    threshold = wrong_scores[int(.9 * (len(wrong_scores) - 1))]
    old_wrong_scores = sorted((fit_wrong @ old_weight).tolist())
    old_threshold = old_wrong_scores[int(.9 * (len(old_wrong_scores) - 1))]
    table = {
        "fit": {"old": summary(correct[fit], fit_wrong, old_weight, old_threshold),
                "separate_head": summary(correct[fit], fit_wrong, weight.detach(), threshold)},
        "development": {"old": summary(correct[dev], dev_wrong, old_weight, old_threshold),
                        "separate_head": summary(correct[dev], dev_wrong,
                                                 weight.detach(), threshold)}}
    baseline = table["development"]["old"]
    candidate = table["development"]["separate_head"]
    development_gate = {
        "pairwise_gain_at_least_5pp": candidate["correct_over_wrong_rate"] >=
            baseline["correct_over_wrong_rate"] + .05,
        "pairwise_rate_at_least_80pct": candidate["correct_over_wrong_rate"] >= .80,
        "correct_acceptance_at_least_50pct": candidate["correct_acceptance"] >= .50,
        "wrong_false_positive_at_most_15pct": candidate["wrong_false_positive"] <= .15}
    audit_gate = None
    if all(development_gate.values()):
        audit = torch.tensor(splits["audit"], dtype=torch.long)
        audit_wrong = wrong_features(splits["audit"])
        table["audit"] = {
            "old": summary(correct[audit], audit_wrong, old_weight, old_threshold),
            "separate_head": summary(correct[audit], audit_wrong,
                                     weight.detach(), threshold)}
        old_a, new_a = table["audit"]["old"], table["audit"]["separate_head"]
        audit_gate = {"pairwise_gain_at_least_5pp":
                      new_a["correct_over_wrong_rate"] >=
                      old_a["correct_over_wrong_rate"] + .05,
                      "pairwise_rate_at_least_75pct":
                      new_a["correct_over_wrong_rate"] >= .75,
                      "correct_acceptance_at_least_50pct":
                      new_a["correct_acceptance"] >= .50,
                      "wrong_false_positive_at_most_15pct":
                      new_a["wrong_false_positive"] <= .15}
    eligible = all(development_gate.values()) and \
        audit_gate is not None and all(audit_gate.values())
    report = {
        "schema": "separate_grounding_head_train_scene_screen_v1",
        "interpretation": "Exploratory train-scene representation screen; success/grounding audit was previously inspected in other probes. No online RL or val-unseen claim.",
        "pair_manifest_sha256": digest(args.pair_manifest),
        "split_manifest_sha256": digest(args.split_manifest),
        "feature_cache_sha256": digest(args.features),
        "old_encoder_sha256": digest(args.encoder),
        "architecture": "Frozen old temporal encoder penultimate 64-dimensional terminal-minus-initial feature; separate linear grounding readout.",
        "training": "Convex fit-scene correct-vs-swapped pairwise softplus with fixed L2 anchor to old scalar head; no development model selection.",
        "fit_wrong_90th_percentile_threshold": threshold,
        "old_fit_wrong_90th_percentile_threshold": old_threshold,
        "weight_displacement_l2": float((weight.detach()-old_weight).norm()),
        "table": table, "development_gate": development_gate,
        "audit_gate": audit_gate, "eligible_for_reward_design": eligible}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if eligible:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"grounding_weight": weight.detach().clone(),
                    "old_encoder_sha256": digest(args.encoder),
                    "pair_manifest_sha256": digest(args.pair_manifest),
                    "split_manifest_sha256": digest(args.split_manifest)},
                   args.checkpoint)
    print(json.dumps({"development": table["development"],
                      "development_gate": development_gate,
                      "audit": table.get("audit"), "audit_gate": audit_gate,
                      "eligible_for_reward_design": eligible}, indent=2))


if __name__ == "__main__":
    main()
