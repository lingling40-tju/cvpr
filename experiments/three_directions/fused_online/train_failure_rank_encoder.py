"""Fine-tune a causal progress encoder on same-goal failed trajectories.

Only fit scenes optimize weights. Development scenes select an epoch. The
scene-disjoint audit partition is never evaluated by this script. Simulator
distance selects pairs but is not a model input or online reward signal.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict, scene_interval


OLD_ENCODER_SHA256 = "1bfb26336b304f9b8a3978070c4fc44b57ae4a3f7717c329d60b5feb845f8b5d"
MODEL_CONFIG_SHA256 = "9e259a64de67f56b23a616e241b8cd130923d2a2759f294b6e1a2471465889f9"
PROMPT_VERSION = "vlnce_server_single_observation_v1"


def evaluate(model: TemporalPotential, hidden: torch.Tensor,
             stop: torch.Tensor, indices: list[int], pairs: list[dict],
             device: torch.device, bootstrap: bool = False) -> dict:
    selected = torch.tensor(indices, dtype=torch.long)
    potentials = predict(model, hidden[selected], device)
    margins = potentials[:, 0, -1] - potentials[:, 1, -1]
    hits = [bool(value > 0) for value in margins]
    scenes = [pairs[index]["scene_id"] for index in indices]
    raw = stop[selected, 0, -1] - stop[selected, 1, -1]
    result = {"pairs": len(indices), "scenes": len(set(scenes)),
              "nearer_failure_above_farther_rate": sum(hits) / len(hits),
              "mean_near_minus_far_margin": float(margins.mean()),
              "raw_stop_reference": float((raw > 0).float().mean())}
    if bootstrap:
        result["scene_bootstrap95"] = scene_interval(hits, scenes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--old-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 60:
        raise ValueError("invalid epoch budget")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    manifest = json.loads(args.manifest.read_text())
    pairs = manifest["pairs"]
    manifest_hash = digest(args.manifest)
    if manifest["schema"] != "failure_rank_train_scene_v1" or \
            len(pairs) != 928 or digest(args.old_encoder) != OLD_ENCODER_SHA256:
        raise ValueError("manifest or frozen encoder mismatch")
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if cache["schema"] != "failure_rank_navigation_sft_state_v1" or \
            cache["manifest_sha256"] != manifest_hash or \
            cache["model_config_sha256"] != MODEL_CONFIG_SHA256 or \
            cache["prompt_version"] != PROMPT_VERSION or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048):
        raise ValueError("incomplete or wrong feature cache")
    for index, pair in enumerate(pairs):
        for role_index, role in enumerate(("near", "far")):
            expected = pair["pair_id"] + "_" + role
            if cache["record_ids"][2 * index + role_index] != expected:
                raise ValueError(f"feature identity mismatch {expected}")
    split = {name: [i for i, pair in enumerate(pairs) if pair["split"] == name]
             for name in ("fit", "development", "audit")}
    if {name: len(indices) for name, indices in split.items()} != manifest["counts"]:
        raise ValueError("split count mismatch")
    if any(set(pairs[i]["scene_id"] for i in split[a]) &
           set(pairs[j]["scene_id"] for j in split[b])
           for a, b in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit"))):
        raise ValueError("scene leakage")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    stop = cache["stop_margin"].float().reshape(-1, 2, 4)
    if not torch.isfinite(hidden).all() or not torch.isfinite(stop).all():
        raise ValueError("nonfinite feature cache")
    old = torch.load(args.old_encoder, map_location="cpu", weights_only=False)
    if old["feature_model_config_sha256"] != MODEL_CONFIG_SHA256:
        raise ValueError("old encoder backbone mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = TemporalPotential().to(device).eval()
    baseline.load_state_dict(old["model"])
    model = TemporalPotential().to(device)
    model.load_state_dict(old["model"])
    baseline_fit = evaluate(baseline, hidden, stop, split["fit"], pairs, device)
    baseline_dev = evaluate(baseline, hidden, stop, split["development"], pairs,
                            device, bootstrap=True)
    teacher = predict(baseline, hidden[split["fit"]], device).to(device)
    fit_index = torch.tensor(split["fit"], dtype=torch.long)
    teacher_by_index = {int(index): teacher[position]
                        for position, index in enumerate(fit_index)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = fit_index[torch.randperm(len(fit_index))]
        losses = []
        for selected in permutation.split(32):
            state = hidden[selected].reshape(-1, 4, 2048).to(device)
            potential = model(state).reshape(-1, 2, 4)
            teacher_batch = torch.stack([teacher_by_index[int(index)]
                                         for index in selected])
            gaps = torch.tensor([pairs[int(index)]["distance_gap_m_for_selection_only"]
                                 for index in selected], device=device)
            target_margin = (gaps / 5.0).clamp(.3, 1.5)
            margin = potential[:, 0, -1] - potential[:, 1, -1]
            ranking = F.softplus(target_margin - margin).mean()
            preserve = F.smooth_l1_loss(potential, teacher_batch, beta=.2)
            loss = ranking + .15 * preserve
            if not torch.isfinite(loss):
                raise ValueError("nonfinite failure-rank loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev = evaluate(model, hidden, stop, split["development"], pairs, device)
        fit = evaluate(model, hidden, stop, split["fit"], pairs, device)
        key = (dev["nearer_failure_above_farther_rate"],
               dev["mean_near_minus_far_margin"], -epoch)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
                        "fit_rate": fit["nearer_failure_above_farther_rate"],
                        "development_rate": dev["nearer_failure_above_farther_rate"]})
        print(json.dumps(history[-1]), flush=True)
        if best is None or key > best[0]:
            best = (key, epoch, {name: value.cpu().clone()
                                 for name, value in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[2])
    final_fit = evaluate(model, hidden, stop, split["fit"], pairs, device,
                         bootstrap=True)
    final_dev = evaluate(model, hidden, stop, split["development"], pairs,
                         device, bootstrap=True)
    gate = {"development_rate_at_least_0_65":
                final_dev["nearer_failure_above_farther_rate"] >= .65,
            "development_gain_over_frozen_v2_at_least_5pp":
                final_dev["nearer_failure_above_farther_rate"] -
                baseline_dev["nearer_failure_above_farther_rate"] >= .05}
    report = {"schema": "failure_rank_encoder_development_v1",
              "interpretation": "Train-scene fit/development only; scene-disjoint audit unopened; no online RL or val-unseen result.",
              "seed": args.seed, "epochs": args.epochs,
              "selected_epoch": best[1], "manifest_sha256": manifest_hash,
              "old_encoder_sha256": OLD_ENCODER_SHA256,
              "model_config_sha256": MODEL_CONFIG_SHA256,
              "baseline_fit": baseline_fit,
              "baseline_development": baseline_dev,
              "fit": final_fit, "development": final_dev,
              "predeclared_gate": gate, "history": history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best[2], "seed": args.seed,
                "selected_epoch": best[1],
                "manifest_sha256": manifest_hash,
                "old_encoder_sha256": OLD_ENCODER_SHA256,
                "model_config_sha256": MODEL_CONFIG_SHA256}, args.checkpoint)
    print(json.dumps({"baseline_development": baseline_dev,
                      "development": final_dev,
                      "predeclared_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
