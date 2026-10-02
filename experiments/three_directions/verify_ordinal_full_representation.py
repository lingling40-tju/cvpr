"""Recompute locked-scene ordinal reward-model diagnostics from checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch

from fit_ordinal_progress_head import ProgressHead, evaluate, load_features


def close(a: float, b: float) -> bool:
    # The report is computed on CUDA and this independent pass on CPU;
    # low-level matmul ordering can move continuous MSE by a few e-6.
    return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)


def bootstrap_scenes(scene_rows: list[dict], field: str, denominator: str,
                     draws: int = 10000) -> list[float]:
    rng = random.Random(20261002)
    values = []
    for _ in range(draws):
        sample = [scene_rows[rng.randrange(len(scene_rows))] for _ in scene_rows]
        total = sum(row[denominator] for row in sample)
        values.append(sum(row[field] * row[denominator] for row in sample) / total)
    values.sort()
    return [values[int(0.025 * draws)], values[int(0.975 * draws)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["fit_scenes"]) != 51 or len(manifest["calibration_scenes"]) != 10:
        raise ValueError("unexpected scene split")
    if set(manifest["fit_scenes"]) & set(manifest["calibration_scenes"]):
        raise ValueError("fit/calibration scene overlap")
    fit = load_features(args.fit_features, manifest, "fit", device)
    if len(fit["episode_ids"]) != 512:
        raise ValueError("incomplete fit episode coverage")
    audit_scenes = manifest["calibration_scenes"][5:]
    audit = load_features(args.calibration_features, manifest, "calibration", device,
                          set(audit_scenes))
    if len(audit["episode_ids"]) == 0:
        raise ValueError("empty locked audit")
    output = {"interpretation": "Locked train-scene representation audit; no RL or val-unseen result.",
              "fit_episodes": 512, "locked_audit_episodes": len(audit["episode_ids"]),
              "locked_audit_scenes": audit_scenes, "seeds": {}, "gate": {}}
    for seed in (11, 22, 33):
        folder = args.run_dir / f"siglip_full_start_relative_seed{seed}"
        report = json.loads((folder / "report.json").read_text())
        checkpoint = torch.load(folder / "head.pt", map_location="cpu", weights_only=False)
        if report["seed"] != checkpoint["seed"] or report["fit_episodes"] != 512 or \
                report["locked_audit_scenes"] != sorted(audit_scenes) or \
                checkpoint["backbone"] != fit["backbone"] or \
                checkpoint["representation"] != "start_relative":
            raise ValueError(f"checkpoint/report mismatch for seed {seed}")
        model = ProgressHead(checkpoint["feature_dim"], checkpoint["representation"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        metrics = evaluate(model, audit)
        for key in ("ordinal_accuracy", "counterfactual_accuracy", "progress_mse",
                    "raw_backbone_ordinal_accuracy", "raw_backbone_counterfactual_accuracy"):
            if not close(metrics[key], report["locked_audit"][key]):
                raise ValueError(f"recomputed {key} differs for seed {seed}")
        scene_rows = []
        for scene in audit_scenes:
            scene_data = load_features(args.calibration_features, manifest, "calibration",
                                       device, {scene})
            scene_metrics = evaluate(model, scene_data)
            scene_rows.append({"scene": scene, **scene_metrics})
        output["seeds"][str(seed)] = {
            "metrics": metrics,
            "scenes": scene_rows,
            "ordinal_scene_bootstrap95": bootstrap_scenes(
                scene_rows, "ordinal_accuracy", "ordered_pairs"),
            "counterfactual_scene_bootstrap95": bootstrap_scenes(
                scene_rows, "counterfactual_accuracy", "counterfactual_comparisons"),
        }
    order = [output["seeds"][str(s)]["metrics"]["ordinal_accuracy"] for s in (11, 22, 33)]
    cf = [output["seeds"][str(s)]["metrics"]["counterfactual_accuracy"] for s in (11, 22, 33)]
    baseline = output["seeds"]["11"]["metrics"]
    if not all(close(output["seeds"][str(s)]["metrics"]["raw_backbone_ordinal_accuracy"],
                     baseline["raw_backbone_ordinal_accuracy"]) and
               close(output["seeds"][str(s)]["metrics"]["raw_backbone_counterfactual_accuracy"],
                     baseline["raw_backbone_counterfactual_accuracy"]) for s in (22, 33)):
        raise ValueError("raw backbone scores differ across head seeds")
    mean_order, mean_cf = statistics.mean(order), statistics.mean(cf)
    conditions = {
        "mean_ordinal_at_least_0_70": mean_order >= 0.70,
        "mean_counterfactual_at_least_0_75": mean_cf >= 0.75,
        "ordinal_gain_over_raw_at_least_0_05":
            mean_order - baseline["raw_backbone_ordinal_accuracy"] >= 0.05,
        "counterfactual_gain_over_raw_at_least_0_05":
            mean_cf - baseline["raw_backbone_counterfactual_accuracy"] >= 0.05,
    }
    output["gate"] = {
        "status": "passed" if all(conditions.values()) else "failed",
        "conditions": conditions,
        "mean_ordinal_accuracy": mean_order,
        "mean_counterfactual_accuracy": mean_cf,
        "mean_ordinal_minus_raw": mean_order - baseline["raw_backbone_ordinal_accuracy"],
        "mean_counterfactual_minus_raw": mean_cf - baseline["raw_backbone_counterfactual_accuracy"],
        "ordinal_seed_sd": statistics.stdev(order),
        "counterfactual_seed_sd": statistics.stdev(cf),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["gate"], indent=2))


if __name__ == "__main__":
    main()
