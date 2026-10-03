"""Fit a compact grounded reward on two frozen temporal representations.

Both encoder scores are computed from cached navigation-SFT states. A small
monotone-input residual head is trained only on fit-scene pair preferences:
near failed > far failed, successful > unsuccessful, and correct instruction
> swapped instruction. Three fixed seeds are averaged. Development scenes
are inspected once after training, with an explicit gate; audit and
val-unseen labels are never used here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


SEEDS = (11, 22, 33)
EPOCHS = 80


class GroundedCalibrator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Positive linear weights preserve each frozen encoder's ordering
        # when the bounded residual is small.
        self.raw_weights = nn.Parameter(torch.full((2,), -0.43275213))
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = F.softplus(self.raw_weights)
        residual = .25 * torch.tanh(self.output(torch.tanh(self.hidden(value))))
        return (value * weights).sum(-1) + residual.squeeze(-1)


def load_encoder(path: Path) -> TemporalPotential:
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoder = TemporalPotential().eval()
    encoder.load_state_dict(state["model"])
    return encoder


def counts(values: torch.Tensor, indices: list[int], scenes: list[str]) -> dict:
    selected = values[indices]
    return {"pairs": len(indices), "scenes": len({scenes[i] for i in indices}),
            "hits": int((selected > 0).sum()),
            "rate": float((selected > 0).float().mean()),
            "mean_margin": float(selected.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("failure_manifest", "failure_features", "old_pair_manifest",
                 "old_v2_manifest", "old_features", "old_swaps", "old_encoder",
                 "new_encoder", "calibration", "output", "checkpoint"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    failure = json.loads(args.failure_manifest.read_text())
    success = json.loads(args.old_pair_manifest.read_text())
    split_manifest = json.loads(args.old_v2_manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    if failure["schema"] != "failure_rank_train_scene_v1" or \
            split_manifest["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            calibration["failure_rank_manifest_sha256"] != digest(args.failure_manifest) or \
            calibration["new_encoder_sha256"] != digest(args.new_encoder) or \
            calibration["old_encoder_sha256"] != digest(args.old_encoder):
        raise ValueError("input provenance mismatch")
    failed_pairs = failure["pairs"]
    success_pairs = success["pairs"]
    old_splits = {row["pair_id"]: row["split"] for row in split_manifest["pairs"]}
    f_idx = {part: [i for i, row in enumerate(failed_pairs) if row["split"] == part]
             for part in ("fit", "development")}
    s_idx = {part: [i for i, row in enumerate(success_pairs)
                    if old_splits[row["pair_id"]] == part]
             for part in ("fit", "development")}
    if [len(f_idx[x]) for x in ("fit", "development")] != [697, 125] or \
            [len(s_idx[x]) for x in ("fit", "development")] != [300, 52]:
        raise ValueError("fit/development coverage mismatch")
    f_cache = torch.load(args.failure_features, map_location="cpu", weights_only=False)
    s_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    if f_cache["manifest_sha256"] != digest(args.failure_manifest) or \
            s_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            f_cache["hidden"].shape != (1856, 4, 2048) or \
            s_cache["hidden"].shape != (800, 4, 2048):
        raise ValueError("feature cache mismatch")
    f_hidden = F.normalize(f_cache["hidden"].float(), dim=-1).reshape(928, 2, 4, 2048)
    s_hidden = F.normalize(s_cache["hidden"].float(), dim=-1).reshape(400, 2, 4, 2048)
    wrong_hidden = torch.zeros(400, 4, 2048)
    for part in ("fit", "development"):
        for i in s_idx[part]:
            pair_id = success_pairs[i]["pair_id"]
            row = torch.load(args.old_swaps / "records" / f"{pair_id}.pt",
                             map_location="cpu", weights_only=False)
            if row["pair_id"] != pair_id or row["hidden"].shape != (4, 2048):
                raise ValueError("wrong-instruction feature mismatch")
            wrong_hidden[i] = F.normalize(row["hidden"].float(), dim=-1)
    af = f_idx["fit"] + f_idx["development"]
    ass = s_idx["fit"] + s_idx["development"]
    scores = {}
    for name, path, scale in (
            ("old", args.old_encoder, float(calibration["old_temporal_scale"])),
            ("new", args.new_encoder, float(calibration["temporal_scale"]))):
        if not math.isfinite(scale) or not 0 < scale < 100:
            raise ValueError("invalid frozen score scale")
        encoder = load_encoder(path)
        scores[name] = {
            "failed": predict(encoder, f_hidden[af], torch.device("cpu"))[:, :, -1] / scale,
            "success": predict(encoder, s_hidden[ass], torch.device("cpu"))[:, :, -1] / scale,
            "wrong": predict(encoder, wrong_hidden[ass].unsqueeze(1),
                             torch.device("cpu"))[:, 0, -1] / scale}
    value = {name: torch.stack((scores["old"][name], scores["new"][name]), dim=-1)
             for name in ("failed", "success", "wrong")}
    f_pos = {index: position for position, index in enumerate(af)}
    s_pos = {index: position for position, index in enumerate(ass)}
    f_fit = [f_pos[i] for i in f_idx["fit"]]
    s_fit = [s_pos[i] for i in s_idx["fit"]]
    f_scenes = [row["scene_id"] for row in failed_pairs]
    s_scenes = [row["scene_id"] for row in success_pairs]

    def evaluate(scored: dict[str, torch.Tensor], part: str) -> dict:
        fm = torch.zeros(928)
        sm = torch.zeros(400)
        gm = torch.zeros(400)
        fm[af] = scored["failed"][:, 0] - scored["failed"][:, 1]
        sm[ass] = scored["success"][:, 0] - scored["success"][:, 1]
        gm[ass] = scored["success"][:, 0] - scored["wrong"]
        return {"failure_near_over_far": counts(fm, f_idx[part], f_scenes),
                "success_over_failure": counts(sm, s_idx[part], s_scenes),
                "correct_over_wrong_instruction": counts(gm, s_idx[part], s_scenes)}

    baselines = {name: {part: evaluate(scores[name], part)
                        for part in ("fit", "development")}
                 for name in ("old", "new")}
    models = []
    histories = []
    for seed in SEEDS:
        random.seed(seed)
        torch.manual_seed(seed)
        calibrator = GroundedCalibrator()
        optimizer = torch.optim.AdamW(calibrator.parameters(), lr=.01,
                                      weight_decay=.02)
        history = []
        for epoch in range(1, EPOCHS + 1):
            calibrator.train()
            f = calibrator(value["failed"][f_fit])
            s = calibrator(value["success"][s_fit])
            w = calibrator(value["wrong"][s_fit])
            failure_loss = F.softplus(.25 - (f[:, 0] - f[:, 1])).mean()
            success_loss = F.softplus(.25 - (s[:, 0] - s[:, 1])).mean()
            grounding_loss = F.softplus(.25 - (s[:, 0] - w)).mean()
            residual_penalty = sum((.25 * torch.tanh(calibrator.output(
                torch.tanh(calibrator.hidden(t))))).square().mean()
                for t in (value["failed"][f_fit].reshape(-1, 2),
                          value["success"][s_fit].reshape(-1, 2)))
            weight_penalty = (F.softplus(calibrator.raw_weights) - .5).square().sum()
            loss = (failure_loss + success_loss + 1.5 * grounding_loss +
                    .02 * residual_penalty + .02 * weight_penalty)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite calibration loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch in (1, 20, 40, 60, 80):
                history.append({"epoch": epoch, "loss": float(loss.detach()),
                                "positive_weights": F.softplus(
                                    calibrator.raw_weights).detach().tolist()})
        calibrator.eval()
        models.append(calibrator)
        histories.append({"seed": seed, "history": history})
    with torch.inference_mode():
        individual = [{name: model(tensor) for name, tensor in value.items()}
                      for model in models]
        ensemble = {name: torch.stack([row[name] for row in individual]).mean(0)
                    for name in value}
    table = {f"seed{seed}": {part: evaluate(row, part)
                             for part in ("fit", "development")}
             for seed, row in zip(SEEDS, individual)}
    table["ensemble"] = {part: evaluate(ensemble, part)
                         for part in ("fit", "development")}
    new_f = baselines["new"]["development"]["failure_near_over_far"]["hits"]
    old_s = baselines["old"]["development"]["success_over_failure"]["hits"]
    old_g = baselines["old"]["development"]["correct_over_wrong_instruction"]["hits"]
    dev = table["ensemble"]["development"]
    gate = {"failure_at_least_3pp_above_new_terminal":
                dev["failure_near_over_far"]["hits"] >= new_f + 4,
            "success_drop_at_most_2pp_from_old_terminal":
                dev["success_over_failure"]["hits"] >= old_s - 1,
            "grounding_drop_at_most_2pp_from_old_terminal":
                dev["correct_over_wrong_instruction"]["hits"] >= old_g - 1}
    report = {"schema": "grounded_progress_calibrator_development_v1",
              "interpretation": "Exploratory train-scene fit/development only. Related development data have been inspected before; neither audit nor val-unseen is opened. No navigation claim.",
              "failure_manifest_sha256": digest(args.failure_manifest),
              "old_v2_manifest_sha256": digest(args.old_v2_manifest),
              "old_encoder_sha256": digest(args.old_encoder),
              "new_encoder_sha256": digest(args.new_encoder),
              "calibration_sha256": digest(args.calibration),
              "seeds": list(SEEDS), "epochs": EPOCHS,
              "loss": "mean softplus(.25-pair_margin) for failed and success pairs, 1.5x grounding, .02x bounded-residual and positive-weight regularization",
              "baselines": baselines, "training": histories,
              "table": table, "development_gate": gate,
              "eligible_development_only": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["eligible_development_only"]:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"states": [model.state_dict() for model in models],
                    "seeds": list(SEEDS), "epochs": EPOCHS,
                    "failure_manifest_sha256": digest(args.failure_manifest),
                    "old_v2_manifest_sha256": digest(args.old_v2_manifest),
                    "old_encoder_sha256": digest(args.old_encoder),
                    "new_encoder_sha256": digest(args.new_encoder)}, args.checkpoint)
    print(json.dumps({"development_gate": gate,
                      "ensemble_development": dev}, indent=2))


if __name__ == "__main__":
    main()
