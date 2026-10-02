"""Train a causal progress representation on cached group-four trajectories.

The train-only geometric teacher supervises intermediate progress, while
same-episode successful/failed rollouts supply a contrastive terminal loss.
The deployable encoder receives only frozen image-instruction SFT states.
Development scenes select the epoch; audit labels are loaded only if its
predeclared gates pass. No online RL is run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_interval(hits: list[bool], scenes: list[str]) -> list[float]:
    unique = sorted(set(scenes))
    by_scene = {scene: [hit for hit, name in zip(hits, scenes) if name == scene]
                for scene in unique}
    rng = random.Random(20261003)
    draws = []
    for _ in range(5000):
        selected = [hit for _ in unique for hit in by_scene[rng.choice(unique)]]
        draws.append(sum(selected) / len(selected))
    draws.sort()
    return [draws[125], draws[4875]]


def evaluate(pred: torch.Tensor, true: torch.Tensor,
             indices: list[int], pairs: list[dict], stop: torch.Tensor,
             bootstrap: bool = True) -> dict:
    selected = torch.tensor(indices, dtype=torch.long)
    endpoint = pred[selected, 0, -1] - pred[selected, 1, -1]
    hits = [bool(x > 0) for x in endpoint]
    scenes = [pairs[i]["scene_id"] for i in indices]
    raw = stop[selected, 0, -1] - stop[selected, 1, -1]
    compared = correct = 0
    for index in indices:
        for role in (0, 1):
            for first in range(4):
                for second in range(first + 1, 4):
                    delta = float(true[index, role, second] - true[index, role, first])
                    if abs(delta) < .02:
                        continue
                    shift = float(pred[index, role, second] - pred[index, role, first])
                    correct += delta * shift > 0
                    compared += 1
    result = {"pairs": len(indices), "scenes": len(set(scenes)),
            "endpoint_success_over_failure": sum(hits) / len(hits),
            "temporal_concordance": correct / compared if compared else None,
            "comparable_frame_pairs": compared,
            "terminal_progress_mae_fraction":
                float((pred[selected, :, -1] - true[selected, :, -1]).abs().mean()),
            "raw_stop_reference": float((raw > 0).float().mean())}
    if bootstrap:
        result["scene_bootstrap95"] = scene_interval(hits, scenes)
    return result


class TemporalPotential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(4096, 128), nn.LayerNorm(128),
                                     nn.GELU(), nn.Dropout(.15))
        self.position = nn.Parameter(torch.zeros(1, 4, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=.15,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64),
                                  nn.GELU(), nn.Linear(64, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [batch, 4, 2048] already L2 normalized.
        reference = hidden[:, :1]
        state = self.project(torch.cat((hidden, hidden - reference), dim=-1))
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool,
                                     device=hidden.device), diagonal=1)
        state = self.temporal(state + self.position, mask=mask)
        scalar = self.head(state).squeeze(-1)
        return scalar - scalar[:, :1]


def predict(model: TemporalPotential, hidden: torch.Tensor,
            device: torch.device) -> torch.Tensor:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in hidden.reshape(-1, 4, 2048).split(64):
            rows.append(model(batch.to(device)).cpu())
    return torch.cat(rows).reshape(-1, 2, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > 100:
        raise ValueError("invalid epoch budget")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    manifest = json.loads(args.manifest.read_text())
    pairs = manifest["pairs"]
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    summary = json.loads((args.labels_root / "summary.json").read_text())
    if cache["schema"] != "navigation_sft_state_v1" or \
            cache["manifest_sha256"] != digest(args.manifest) or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048) or \
            summary["manifest_sha256"] != digest(args.manifest) or \
            summary["completed_trajectories"] != 2 * len(pairs) or summary["errors"]:
        raise ValueError("incomplete frozen feature/label coverage")
    split = {name: [i for i, pair in enumerate(pairs) if pair["split"] == name]
             for name in ("fit", "development", "audit")}
    if {name: len(rows) for name, rows in split.items()} != manifest["counts"]:
        raise ValueError("scene split mismatch")
    if any(set(pairs[i]["scene_id"] for i in split[a]) &
           set(pairs[j]["scene_id"] for j in split[b])
           for a, b in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit"))):
        raise ValueError("scene leakage")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    stop = cache["stop_margin"].float().reshape(-1, 2, 4)
    truth = torch.zeros(len(pairs), 2, 4)
    def read_labels(indices: list[int]) -> None:
        for index in indices:
            pair = pairs[index]
            for role_index, role in enumerate(("success", "failure")):
                record_id = pair["pair_id"] + "_" + role
                if cache["record_ids"][2 * index + role_index] != record_id:
                    raise ValueError(f"feature identity mismatch {record_id}")
                data = json.loads((args.labels_root / "records" /
                                   f"{record_id}.json").read_text())
                if data["record_id"] != record_id or data["split"] != pair["split"]:
                    raise ValueError(f"label identity mismatch {record_id}")
                distances = data["distance_to_goal_m"]
                if len(distances) != 4 or distances[0] <= 0:
                    raise ValueError(f"invalid label {record_id}")
                truth[index, role_index] = torch.tensor(
                    [(distances[0] - distance) / distances[0]
                     for distance in distances])
    read_labels(split["fit"] + split["development"])
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
        for pair_batch in permutation.split(32):
            observations = hidden[pair_batch].reshape(-1, 4, 2048).to(device)
            labels = truth[pair_batch].reshape(-1, 4).to(device).clamp(-2, 1.5)
            potential = model(observations).reshape(-1, 2, 4)
            targets = labels.reshape(-1, 2, 4)
            geometry = F.smooth_l1_loss(potential[:, :, 1:],
                                        targets[:, :, 1:], beta=.2)
            terminal_gap = potential[:, 0, -1] - potential[:, 1, -1]
            ranking = F.softplus(.3 - terminal_gap).mean()
            delta_true = targets[:, :, 1:] - targets[:, :, :-1]
            delta_pred = potential[:, :, 1:] - potential[:, :, :-1]
            valid = delta_true.abs() >= .03
            ordering = F.softplus(-delta_true.sign() * delta_pred)
            ordering = (ordering * valid).sum() / valid.sum().clamp_min(1)
            loss = geometry + .35 * ranking + .15 * ordering
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predicted = predict(model, hidden, device)
        dev = evaluate(predicted, truth, split["development"], pairs, stop,
                       bootstrap=False)
        fit = evaluate(predicted, truth, split["fit"], pairs, stop,
                       bootstrap=False)
        key = (dev["endpoint_success_over_failure"],
               dev["temporal_concordance"], -dev["terminal_progress_mae_fraction"],
               -epoch)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
                        "fit_endpoint": fit["endpoint_success_over_failure"],
                        "development_endpoint": dev["endpoint_success_over_failure"],
                        "development_temporal": dev["temporal_concordance"]})
        print(json.dumps(history[-1]), flush=True)
        if best is None or key > best[0]:
            best = (key, epoch, {name: value.cpu().clone()
                                 for name, value in model.state_dict().items()})
    assert best is not None
    model.load_state_dict(best[2])
    predicted = predict(model, hidden, device)
    dev = evaluate(predicted, truth, split["development"], pairs, stop)
    fit = evaluate(predicted, truth, split["fit"], pairs, stop)
    gate = {"development_endpoint_at_least_0_70":
                dev["endpoint_success_over_failure"] >= .70,
            "development_temporal_at_least_0_60":
                dev["temporal_concordance"] >= .60}
    audit = None
    if all(gate.values()):
        read_labels(split["audit"])
        audit = evaluate(predicted, truth, split["audit"], pairs, stop)
        gate.update({"audit_endpoint_at_least_0_75":
                         audit["endpoint_success_over_failure"] >= .75,
                     "audit_gain_over_raw_stop_at_least_5pp":
                         audit["endpoint_success_over_failure"] -
                         audit["raw_stop_reference"] >= .05,
                     "audit_temporal_at_least_0_60":
                         audit["temporal_concordance"] >= .60})
    report = {"schema": "temporal_progress_encoder_screen_v1",
              "interpretation": "Train-scene adapter screen; no RL or val-unseen result.",
              "seed": args.seed, "epochs": args.epochs, "selected_epoch": best[1],
              "manifest_sha256": digest(args.manifest),
              "feature_model_config_sha256": cache["model_config_sha256"],
              "feature_prompt_version": cache["prompt_version"],
              "label_summary_sha256": digest(args.labels_root / "summary.json"),
              "architecture": "1-layer causal transformer over four SFT image-instruction states",
              "loss": "train-only geodesic progress + same-episode contrastive ranking + temporal ordering",
              "fit": fit, "development": dev, "audit": audit,
              "predeclared_gate": gate, "history": history}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best[2], "seed": args.seed,
                "selected_epoch": best[1],
                "manifest_sha256": digest(args.manifest),
                "feature_model_config_sha256": cache["model_config_sha256"]},
               args.checkpoint)
    print(json.dumps({"selected_epoch": best[1], "development": dev,
                      "audit": audit, "predeclared_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
