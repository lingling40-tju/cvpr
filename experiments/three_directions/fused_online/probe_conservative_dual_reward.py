"""Screen two fixed, conservative combinations of old and new potentials.

Only fit/development train scenes are read. The script never opens either
audit partition or uses val-unseen navigation episodes for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


def load_model(path: Path) -> TemporalPotential:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalPotential().eval()
    model.load_state_dict(checkpoint["model"])
    return model


def scores(model: TemporalPotential, features: torch.Tensor,
           indices: list[int], scale: float) -> torch.Tensor:
    result = torch.zeros(features.shape[:2])
    result[indices] = torch.sigmoid(
        predict(model, features[indices], torch.device("cpu"))[:, :, -1] / scale)
    return result


def summary(margins: torch.Tensor, indices: list[int], scenes: list[str]) -> dict:
    selected = margins[indices]
    return {"pairs": len(indices),
            "scenes": len({scenes[i] for i in indices}),
            "hits": int((selected > 0).sum()),
            "rate": float((selected > 0).float().mean()),
            "mean_margin": float(selected.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("failure_manifest", "failure_features", "old_pair_manifest",
                 "old_v2_manifest", "old_features", "old_swaps", "old_encoder",
                 "new_encoder", "calibration", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    failure = json.loads(args.failure_manifest.read_text())
    older = json.loads(args.old_pair_manifest.read_text())
    older_split = json.loads(args.old_v2_manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    if failure["schema"] != "failure_rank_train_scene_v1" or \
            older_split["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            calibration["failure_rank_manifest_sha256"] != digest(args.failure_manifest) or \
            calibration["new_encoder_sha256"] != digest(args.new_encoder) or \
            calibration["old_encoder_sha256"] != digest(args.old_encoder):
        raise ValueError("input provenance mismatch")
    failed_pairs = failure["pairs"]
    success_pairs = older["pairs"]
    split_by_id = {row["pair_id"]: row["split"] for row in older_split["pairs"]}
    if len(failed_pairs) != 928 or len(success_pairs) != 400:
        raise ValueError("unexpected pair coverage")
    failed_idx = {name: [i for i, row in enumerate(failed_pairs)
                         if row["split"] == name]
                  for name in ("fit", "development")}
    success_idx = {name: [i for i, row in enumerate(success_pairs)
                          if split_by_id[row["pair_id"]] == name]
                   for name in ("fit", "development")}
    if [len(failed_idx[x]) for x in ("fit", "development")] != [697, 125] or \
            [len(success_idx[x]) for x in ("fit", "development")] != [300, 52]:
        raise ValueError("split coverage mismatch")
    failed_cache = torch.load(args.failure_features, map_location="cpu", weights_only=False)
    success_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    if failed_cache["manifest_sha256"] != digest(args.failure_manifest) or \
            success_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            failed_cache["hidden"].shape != (1856, 4, 2048) or \
            success_cache["hidden"].shape != (800, 4, 2048):
        raise ValueError("feature cache mismatch")
    failed_hidden = F.normalize(failed_cache["hidden"].float(), dim=-1).reshape(928, 2, 4, 2048)
    success_hidden = F.normalize(success_cache["hidden"].float(), dim=-1).reshape(400, 2, 4, 2048)
    # Wrong-instruction features are loaded only for fit/development pairs.
    wrong_hidden = torch.zeros(400, 1, 4, 2048)
    for split in ("fit", "development"):
        for index in success_idx[split]:
            pair_id = success_pairs[index]["pair_id"]
            item = torch.load(args.old_swaps / "records" / f"{pair_id}.pt",
                              map_location="cpu", weights_only=False)
            if item["pair_id"] != pair_id or item["hidden"].shape != (4, 2048):
                raise ValueError("wrong-instruction feature mismatch")
            wrong_hidden[index, 0] = F.normalize(item["hidden"].float(), dim=-1)
    old = load_model(args.old_encoder)
    new = load_model(args.new_encoder)
    scales = {"old": float(calibration["old_temporal_scale"]),
              "new": float(calibration["temporal_scale"])}
    for scale in scales.values():
        if not 0 < scale < 100:
            raise ValueError("invalid fit-only score scale")
    active_f = failed_idx["fit"] + failed_idx["development"]
    active_s = success_idx["fit"] + success_idx["development"]
    predictions = {}
    for name, model in (("old", old), ("new", new)):
        predictions[name] = {
            "failed": scores(model, failed_hidden, active_f, scales[name]),
            "success": scores(model, success_hidden, active_s, scales[name]),
            "wrong": scores(model, wrong_hidden, active_s, scales[name])[:, 0]}
    combinations = {
        "old": lambda a, b: a,
        "new": lambda a, b: b,
        "product": lambda a, b: a * b,
        "minimum": lambda a, b: torch.minimum(a, b),
    }
    failed_scenes = [row["scene_id"] for row in failed_pairs]
    success_scenes = [row["scene_id"] for row in success_pairs]
    table = {}
    for name, combine in combinations.items():
        f = combine(predictions["old"]["failed"], predictions["new"]["failed"])
        s = combine(predictions["old"]["success"], predictions["new"]["success"])
        w = combine(predictions["old"]["wrong"], predictions["new"]["wrong"])
        failed_margin = f[:, 0] - f[:, 1]
        success_margin = s[:, 0] - s[:, 1]
        grounding_margin = s[:, 0] - w
        table[name] = {split: {
            "failure_near_over_far": summary(failed_margin, failed_idx[split], failed_scenes),
            "success_over_failure": summary(success_margin, success_idx[split], success_scenes),
            "correct_over_wrong_instruction": summary(grounding_margin, success_idx[split], success_scenes)}
                       for split in ("fit", "development")}
    base_new = table["new"]["development"]
    base_old = table["old"]["development"]
    for name in ("product", "minimum"):
        dev = table[name]["development"]
        table[name]["development_gate"] = {
            "failure_gain_at_least_3pp_vs_new":
                dev["failure_near_over_far"]["rate"] >=
                base_new["failure_near_over_far"]["rate"] + .03 - 1e-8,
            "success_drop_at_most_2pp_vs_old":
                dev["success_over_failure"]["rate"] >=
                base_old["success_over_failure"]["rate"] - .02 - 1e-8,
            "grounding_drop_at_most_2pp_vs_old":
                dev["correct_over_wrong_instruction"]["rate"] >=
                base_old["correct_over_wrong_instruction"]["rate"] - .02 - 1e-8}
    eligible = [name for name in ("product", "minimum")
                if all(table[name]["development_gate"].values())]
    report = {"schema": "conservative_dual_reward_development_v1",
              "interpretation": "Train-scene fit/development probe only. Both audit partitions and val-unseen remain unopened. No navigation claim.",
              "failure_manifest_sha256": digest(args.failure_manifest),
              "old_v2_manifest_sha256": digest(args.old_v2_manifest),
              "old_encoder_sha256": digest(args.old_encoder),
              "new_encoder_sha256": digest(args.new_encoder),
              "calibration_sha256": digest(args.calibration),
              "formulas": {"product": "sigmoid(old/old_fit_scale)*sigmoid(new/new_fit_scale)",
                           "minimum": "min(sigmoid(old/old_fit_scale),sigmoid(new/new_fit_scale))"},
              "table": table, "eligible_development_only": eligible}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"eligible_development_only": eligible,
                      "development": {name: row["development"] for name, row in table.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
