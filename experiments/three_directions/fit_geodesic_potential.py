"""Fit a small navigation-SFT progress potential from train-only geometry.

The model sees frozen image-instruction hidden states, never privileged
distance at inference. Fit scenes train a ridge potential, development scenes
choose regularization, and audit scenes are read once for the frozen choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_interval(hits: list[bool], scenes: list[str]) -> list[float]:
    unique = sorted(set(scenes))
    grouped = {scene: [hit for hit, name in zip(hits, scenes) if name == scene]
               for scene in unique}
    rng = random.Random(20261003)
    draws = []
    for _ in range(5000):
        selected = [value for _ in unique for value in grouped[rng.choice(unique)]]
        draws.append(sum(selected) / len(selected))
    draws.sort()
    return [draws[125], draws[4875]]


def temporal_concordance(predicted: torch.Tensor, true: torch.Tensor) -> dict:
    hit, total = 0, 0
    for row_pred, row_true in zip(predicted, true):
        for first in range(4):
            for second in range(first + 1, 4):
                delta_true = float(row_true[second] - row_true[first])
                if abs(delta_true) < .02:
                    continue
                delta_pred = float(row_pred[second] - row_pred[first])
                hit += (delta_true * delta_pred > 0)
                total += 1
    return {"comparable_frame_pairs": total,
            "fraction_correct": hit / total if total else None}


def score_split(indices: list[int], prediction: torch.Tensor,
                truth: torch.Tensor, scenes_all: list[str],
                stop_margin: torch.Tensor) -> dict:
    trajectory_indices = [i for pair in indices for i in (2 * pair, 2 * pair + 1)]
    pred = prediction[trajectory_indices]
    actual = truth[trajectory_indices]
    differences = prediction[2 * torch.tensor(indices), -1] - \
        prediction[2 * torch.tensor(indices) + 1, -1]
    hits = [bool(value > 0) for value in differences]
    scenes = [scenes_all[i] for i in indices]
    raw = stop_margin[2 * torch.tensor(indices), -1] - \
        stop_margin[2 * torch.tensor(indices) + 1, -1]
    result = {"pairs": len(indices), "scenes": len(set(scenes)),
              "endpoint_success_over_failure": sum(hits) / len(hits),
              "scene_bootstrap95": scene_interval(hits, scenes),
              "previous_raw_stop_ranking": float((raw > 0).float().mean()),
              "terminal_progress_mae_fraction": float((pred[:, -1] - actual[:, -1]).abs().mean()),
              "temporal_concordance": temporal_concordance(pred, actual)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    features = torch.load(args.features, map_location="cpu", weights_only=False)
    label_summary = json.loads((args.labels_root / "summary.json").read_text())
    pairs = manifest["pairs"]
    if features["schema"] != "navigation_sft_state_v1" or \
            features["manifest_sha256"] != digest(args.manifest) or \
            features["hidden"].shape != (2 * len(pairs), 4, 2048) or \
            label_summary["manifest_sha256"] != digest(args.manifest) or \
            label_summary["completed_trajectories"] != 2 * len(pairs) or \
            label_summary["errors"]:
        raise ValueError("cached features/labels are incomplete")
    targets = []
    for pair in pairs:
        for role in ("success", "failure"):
            record_id = pair["pair_id"] + "_" + role
            if features["record_ids"][len(targets)] != record_id:
                raise ValueError(f"feature mismatch {record_id}")
            if pair["split"] == "audit":
                targets.append([0.0] * 4)  # Do not read audit labels before the dev gate.
                continue
            label = json.loads((args.labels_root / "records" /
                                f"{record_id}.json").read_text())
            if label["record_id"] != record_id or label["split"] != pair["split"]:
                raise ValueError(f"label mismatch {record_id}")
            distance = label["distance_to_goal_m"]
            if len(distance) != 4 or distance[0] <= 0:
                raise ValueError(f"invalid geodesic label {record_id}")
            targets.append([float((distance[0] - d) / distance[0]) for d in distance])
    y = torch.tensor(targets, dtype=torch.float32)
    h = F.normalize(features["hidden"].float(), dim=-1)
    x = h - h[:, :1]
    scenes = [pair["scene_id"] for pair in pairs]
    split = {name: [i for i, pair in enumerate(pairs) if pair["split"] == name]
             for name in ("fit", "development", "audit")}
    if {name: len(indices) for name, indices in split.items()} != manifest["counts"]:
        raise ValueError("scene split count mismatch")
    if any(set(scenes[i] for i in split[a]) & set(scenes[j] for j in split[b])
           for a, b in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit"))):
        raise ValueError("scene leakage")
    fit_trajectories = [j for i in split["fit"] for j in (2 * i, 2 * i + 1)]
    train_x = x[fit_trajectories, 1:].reshape(-1, x.shape[-1]).cuda()
    train_y = y[fit_trajectories, 1:].reshape(-1).cuda()
    gram = train_x @ train_x.T
    identity = torch.eye(len(gram), device=gram.device)
    candidates = []
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        weight = train_x.T @ torch.linalg.solve(gram + alpha * identity, train_y)
        predicted = (x.reshape(-1, x.shape[-1]).cuda() @ weight).cpu().reshape(-1, 4)
        dev = score_split(split["development"], predicted, y, scenes,
                          features["stop_margin"].float())
        candidates.append((dev["endpoint_success_over_failure"],
                           dev["temporal_concordance"]["fraction_correct"],
                           -alpha, alpha, weight.cpu(), dev))
    chosen = max(candidates, key=lambda row: row[:3])
    _, _, _, alpha, weight, development = chosen
    prediction = (x.reshape(-1, x.shape[-1]).cuda() @ weight.cuda()).cpu().reshape(-1, 4)
    fit = score_split(split["fit"], prediction, y, scenes,
                      features["stop_margin"].float())
    gate = {"development_endpoint_at_least_0_70":
                development["endpoint_success_over_failure"] >= .70,
            "development_temporal_at_least_0_60":
                development["temporal_concordance"]["fraction_correct"] >= .60}
    audit = None
    if all(gate.values()):
        for pair_index in split["audit"]:
            pair = pairs[pair_index]
            for role_offset, role in enumerate(("success", "failure")):
                record_id = pair["pair_id"] + "_" + role
                label = json.loads((args.labels_root / "records" /
                                    f"{record_id}.json").read_text())
                if label["record_id"] != record_id or label["split"] != "audit":
                    raise ValueError(f"audit label mismatch {record_id}")
                distance = label["distance_to_goal_m"]
                if len(distance) != 4 or distance[0] <= 0:
                    raise ValueError(f"invalid audit geodesic label {record_id}")
                y[2 * pair_index + role_offset] = torch.tensor(
                    [(distance[0] - d) / distance[0] for d in distance])
        audit = score_split(split["audit"], prediction, y, scenes,
                            features["stop_margin"].float())
        gate.update({"audit_endpoint_at_least_0_75":
                         audit["endpoint_success_over_failure"] >= .75,
                     "audit_gain_over_raw_stop_at_least_5pp":
                         audit["endpoint_success_over_failure"] -
                         audit["previous_raw_stop_ranking"] >= .05,
                     "audit_temporal_at_least_0_60":
                         audit["temporal_concordance"]["fraction_correct"] >= .60})
    report = {"schema": "geodesic_potential_audit_v1",
              "interpretation": "Train-scene potential screen; no RL or val-unseen result.",
              "manifest_sha256": digest(args.manifest),
              "feature_prompt_version": features["prompt_version"],
              "feature_model_config_sha256": features["model_config_sha256"],
              "label_summary_sha256": digest(args.labels_root / "summary.json"),
              "representation": "L2-normalized navigation-SFT hidden state difference from start",
              "target": "geodesic distance reduction divided by initial distance",
              "regularization": alpha,
              "fit": fit, "development": development, "audit": audit,
              "development_candidates": [
                  {"alpha": row[3], "endpoint_accuracy": row[5]["endpoint_success_over_failure"],
                   "temporal_concordance": row[5]["temporal_concordance"]["fraction_correct"]}
                  for row in candidates],
              "predeclared_gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight, "regularization": alpha,
                "manifest_sha256": digest(args.manifest),
                "feature_model_config_sha256": features["model_config_sha256"],
                "label_summary_sha256": digest(args.labels_root / "summary.json")},
               args.weights_output)
    print(json.dumps({"development": development, "audit": audit,
                      "predeclared_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
