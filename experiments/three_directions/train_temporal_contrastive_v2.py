"""Train a causal progress encoder with hard wrong-instruction negatives.

This phase uses only fit/development scenes from a newly frozen split. It
selects an epoch by balanced development gates, saves weights, and never
reads the fresh audit labels or wrong-instruction features.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, evaluate, predict, scene_interval


def instruction_score(correct: torch.Tensor, wrong: torch.Tensor,
                      indices: list[int], pairs: list[dict],
                      bootstrap: bool = True) -> dict:
    hits = [bool(correct[i, 0, -1] > wrong[i, -1]) for i in indices]
    scenes = [pairs[i]["scene_id"] for i in indices]
    margins = [float(correct[i, 0, -1] - wrong[i, -1]) for i in indices]
    result = {"pairs": len(indices), "scenes": len(set(scenes)),
              "correct_instruction_preference": sum(hits) / len(hits),
              "mean_correct_minus_wrong": sum(margins) / len(margins)}
    if bootstrap:
        result["scene_bootstrap95"] = scene_interval(hits, scenes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--swaps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        raise ValueError("invalid epoch budget")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    source = json.loads(args.manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    pairs = source["pairs"]
    if v2["source_manifest_sha256"] != digest(args.manifest) or \
            v2["group_size"] != 4 or len(v2["pairs"]) != len(pairs):
        raise ValueError("source/v2 mismatch")
    rows = {row["pair_id"]: row for row in v2["pairs"]}
    split = {name: [i for i, pair in enumerate(pairs)
                    if rows[pair["pair_id"]]["split"] == name]
             for name in ("fit", "development", "audit")}
    if {name: len(ids) for name, ids in split.items()} != v2["counts"]:
        raise ValueError("v2 split count mismatch")
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    labels = json.loads((args.labels_root / "summary.json").read_text())
    if cache["manifest_sha256"] != digest(args.manifest) or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048) or \
            labels["manifest_sha256"] != digest(args.manifest) or \
            labels["completed_trajectories"] != 2 * len(pairs) or labels["errors"]:
        raise ValueError("incomplete frozen feature/geometry coverage")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    stop = cache["stop_margin"].float().reshape(-1, 2, 4)
    truth = torch.zeros(len(pairs), 2, 4)
    wrong = torch.zeros(len(pairs), 4, 2048)
    v2_hash = digest(args.v2_manifest)
    for index in split["fit"] + split["development"]:
        pair = pairs[index]
        for role_index, role in enumerate(("success", "failure")):
            record_id = pair["pair_id"] + "_" + role
            if cache["record_ids"][2 * index + role_index] != record_id:
                raise ValueError(f"feature identity mismatch {record_id}")
            item = json.loads((args.labels_root / "records" /
                               f"{record_id}.json").read_text())
            if item["record_id"] != record_id:
                raise ValueError(f"geometry identity mismatch {record_id}")
            distance = item["distance_to_goal_m"]
            if len(distance) != 4 or distance[0] <= 0:
                raise ValueError(f"invalid distance {record_id}")
            truth[index, role_index] = torch.tensor(
                [(distance[0] - d) / distance[0] for d in distance])
        swap = torch.load(args.swaps_root / "records" /
                          f"{pair['pair_id']}.pt", map_location="cpu",
                          weights_only=False)
        if swap["pair_id"] != pair["pair_id"] or \
                swap["manifest_sha256"] != v2_hash or \
                swap["model_config_sha256"] != cache["model_config_sha256"] or \
                swap["hidden"].shape != (4, 2048):
            raise ValueError(f"wrong-instruction feature mismatch {pair['pair_id']}")
        wrong[index] = F.normalize(swap["hidden"].float(), dim=-1)
    device = torch.device("cuda:0")
    model = TemporalPotential().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4,
                                  weight_decay=.02)
    fit_indices = torch.tensor(split["fit"], dtype=torch.long)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = fit_indices[torch.randperm(len(fit_indices))]
        losses = []
        for selected in permutation.split(32):
            state = hidden[selected].reshape(-1, 4, 2048).to(device)
            wrong_state = wrong[selected].to(device)
            targets = truth[selected].to(device).clamp(-2, 1.5)
            potential = model(state).reshape(-1, 2, 4)
            wrong_potential = model(wrong_state)
            geometry = F.smooth_l1_loss(potential[:, :, 1:],
                                        targets[:, :, 1:], beta=.2)
            endpoint_gap = potential[:, 0, -1] - potential[:, 1, -1]
            ranking = F.softplus(.3 - endpoint_gap).mean()
            delta_true = targets[:, :, 1:] - targets[:, :, :-1]
            delta_pred = potential[:, :, 1:] - potential[:, :, :-1]
            valid = delta_true.abs() >= .03
            ordering = (F.softplus(-delta_true.sign() * delta_pred) * valid).sum() / \
                valid.sum().clamp_min(1)
            instruction = F.softplus(
                .2 - (potential[:, 0, -1] - wrong_potential[:, -1])).mean()
            loss = geometry + .35 * ranking + .15 * ordering + .5 * instruction
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        correct_pred = predict(model, hidden, device)
        wrong_pred = predict(model, wrong.unsqueeze(1), device)[:, 0]
        dev = evaluate(correct_pred, truth, split["development"], pairs,
                       stop, bootstrap=False)
        grounded = instruction_score(correct_pred, wrong_pred,
                                     split["development"], pairs, bootstrap=False)
        fit = evaluate(correct_pred, truth, split["fit"], pairs,
                       stop, bootstrap=False)
        # Select the epoch that best clears all three development margins.
        key = (min(dev["endpoint_success_over_failure"] - .70,
                   dev["temporal_concordance"] - .60,
                   grounded["correct_instruction_preference"] - .75),
               dev["endpoint_success_over_failure"] +
               grounded["correct_instruction_preference"],
               dev["temporal_concordance"], -epoch)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
                        "fit_endpoint": fit["endpoint_success_over_failure"],
                        "development_endpoint": dev["endpoint_success_over_failure"],
                        "development_temporal": dev["temporal_concordance"],
                        "development_instruction": grounded["correct_instruction_preference"]})
        print(json.dumps(history[-1]), flush=True)
        if best is None or key > best[0]:
            best = (key, epoch, {name: value.cpu().clone()
                                 for name, value in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[2])
    correct_pred = predict(model, hidden, device)
    wrong_pred = predict(model, wrong.unsqueeze(1), device)[:, 0]
    development = evaluate(correct_pred, truth, split["development"], pairs, stop)
    grounding = instruction_score(correct_pred, wrong_pred,
                                  split["development"], pairs)
    fit = evaluate(correct_pred, truth, split["fit"], pairs, stop)
    fit_grounding = instruction_score(correct_pred, wrong_pred, split["fit"], pairs)
    gate = {"development_endpoint_at_least_0_70":
                development["endpoint_success_over_failure"] >= .70,
            "development_temporal_at_least_0_60":
                development["temporal_concordance"] >= .60,
            "development_instruction_at_least_0_75":
                grounding["correct_instruction_preference"] >= .75}
    report = {"schema": "temporal_contrastive_v2_development_v1",
              "interpretation": "Fresh train-scene development screen; audit not read; no RL.",
              "seed": args.seed, "epochs": args.epochs, "selected_epoch": best[1],
              "source_manifest_sha256": digest(args.manifest),
              "v2_manifest_sha256": v2_hash,
              "feature_model_config_sha256": cache["model_config_sha256"],
              "feature_prompt_version": cache["prompt_version"],
              "fit": fit, "fit_grounding": fit_grounding,
              "development": development,
              "development_grounding": grounding,
              "predeclared_gate": gate,
              "history": history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best[2], "seed": args.seed,
                "selected_epoch": best[1],
                "source_manifest_sha256": digest(args.manifest),
                "v2_manifest_sha256": v2_hash,
                "feature_model_config_sha256": cache["model_config_sha256"]},
               args.checkpoint)
    print(json.dumps({"selected_epoch": best[1], "development": development,
                      "development_grounding": grounding,
                      "predeclared_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
