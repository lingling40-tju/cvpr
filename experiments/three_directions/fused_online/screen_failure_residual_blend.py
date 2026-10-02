"""Select a conservative old/new temporal blend using development scenes only.

The new representation was trained for failed-pair ranking.  A normalized
residual blend keeps the older success/grounding representation as an anchor.
This screen never uses failure-rank audit pairs to choose the blend weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_model(path: Path, device: torch.device) -> TemporalPotential:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalPotential().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def split_accuracy(margins: torch.Tensor, pairs: list[dict],
                   split: str) -> dict:
    indices = [i for i, pair in enumerate(pairs) if pair["split"] == split]
    if not indices:
        raise ValueError(f"empty {split} split")
    selected = margins[indices]
    return {"pairs": len(indices), "scenes": len({pairs[i]["scene_id"] for i in indices}),
            "hits": int((selected > 0).sum()),
            "rate": float((selected > 0).float().mean()),
            "mean_margin": float(selected.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--failure-features", type=Path, required=True)
    parser.add_argument("--old-pair-manifest", type=Path, required=True)
    parser.add_argument("--old-v2-manifest", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--old-encoder", type=Path, required=True)
    parser.add_argument("--new-encoder", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    failure = json.loads(args.failure_manifest.read_text())
    old_pairs = json.loads(args.old_pair_manifest.read_text())["pairs"]
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    if failure["schema"] != "failure_rank_train_scene_v1" or \
            old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            calibration["failure_rank_manifest_sha256"] != digest(args.failure_manifest) or \
            calibration["new_encoder_sha256"] != digest(args.new_encoder) or \
            calibration["old_encoder_sha256"] != digest(args.old_encoder):
        raise ValueError("screen provenance mismatch")
    old_split = {row["pair_id"]: row["split"] for row in old_v2["pairs"]}
    success_pairs = [{"pair_id": pair["pair_id"], "scene_id": pair["scene_id"],
                      "split": old_split[pair["pair_id"]]} for pair in old_pairs]
    failure_pairs = failure["pairs"]
    old_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    failure_cache = torch.load(args.failure_features, map_location="cpu", weights_only=False)
    if old_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            failure_cache["manifest_sha256"] != digest(args.failure_manifest) or \
            old_cache["hidden"].shape != (2 * len(success_pairs), 4, 2048) or \
            failure_cache["hidden"].shape != (2 * len(failure_pairs), 4, 2048):
        raise ValueError("incomplete feature cache")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    old = load_model(args.old_encoder, device)
    new = load_model(args.new_encoder, device)
    old_scale = calibration["old_temporal_scale"]
    new_scale = calibration["temporal_scale"]

    def normalized_pair_margins(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
        old_pred = predict(old, hidden, device)
        new_pred = predict(new, hidden, device)
        return ((old_pred[:, 0, -1] - old_pred[:, 1, -1]) / old_scale,
                (new_pred[:, 0, -1] - new_pred[:, 1, -1]) / new_scale)

    old_success, new_success = normalized_pair_margins(old_cache)
    old_failure, new_failure = normalized_pair_margins(failure_cache)
    table = []
    for weight in WEIGHTS:
        success = (1 - weight) * old_success + weight * new_success
        failed = (1 - weight) * old_failure + weight * new_failure
        table.append({"new_weight": weight,
                      "success_fit": split_accuracy(success, success_pairs, "fit"),
                      "success_development": split_accuracy(success, success_pairs, "development"),
                      "failed_fit": split_accuracy(failed, failure_pairs, "fit"),
                      "failed_development": split_accuracy(failed, failure_pairs, "development")})
    base = table[0]
    for row in table:
        row["selection_gate"] = {
            "failed_development_gain_at_least_5pp":
                row["failed_development"]["rate"] -
                base["failed_development"]["rate"] >= .05 - 1e-8,
            "success_fit_drop_at_most_2pp":
                row["success_fit"]["rate"] >= base["success_fit"]["rate"] - .02 - 1e-8,
            "success_development_drop_at_most_2pp":
                row["success_development"]["rate"] >=
                base["success_development"]["rate"] - .02 - 1e-8}
    eligible = [row for row in table if all(row["selection_gate"].values())]
    selected = min(eligible, key=lambda row: row["new_weight"])["new_weight"] if eligible else None
    failure_fit_scenes = {row["scene_id"] for row in failure_pairs
                          if row["split"] == "fit"}
    success_scene_overlap = {
        split: len(failure_fit_scenes & {row["scene_id"] for row in success_pairs
                                         if row["split"] == split})
        for split in ("fit", "development", "audit")}
    report = {"schema": "failure_residual_blend_development_v1",
              "interpretation": "Weight chosen only on train-scene fit/development pairs. The older success-pair development scenes overlap the new failure-rank fit scenes. No failure audit, online RL, or val-unseen navigation result.",
              "failure_rank_manifest_sha256": digest(args.failure_manifest),
              "old_v2_manifest_sha256": digest(args.old_v2_manifest),
              "calibration_sha256": digest(args.calibration),
              "success_scenes_overlapping_failure_fit": success_scene_overlap,
              "weights": list(WEIGHTS), "table": table, "selected_weight": selected,
              "selection_rule": "smallest positive weight meeting >=5pp failed-pair development gain and <=2pp success-pair fit/development drops"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
