"""Exploratory frozen-encoder agreement screen on train-scene failure pairs.

The old encoder retained instruction grounding; the new encoder ranked
failed navigation endpoints better. This fixed rule keeps a failure pair
only when both encoders agree on its ordering. Scene splits and all inputs
were created before this probe, but the earlier audit has been inspected,
so the result is exploratory rather than an independent validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, predict


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(path: Path) -> TemporalPotential:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TemporalPotential().eval()
    model.load_state_dict(checkpoint["model"])
    return model


def summarize(indices: list[int], old_delta: torch.Tensor,
              new_delta: torch.Tensor, scenes: list[str]) -> dict:
    new = new_delta[indices]
    old = old_delta[indices]
    selected = (new != 0) & (old != 0) & (new.sign() == old.sign())
    hits = int((new > 0).sum())
    selected_count = int(selected.sum())
    selected_hits = int(((new > 0) & selected).sum())
    return {"pairs": len(indices), "scenes": len({scenes[i] for i in indices}),
            "new_hits_all": hits, "new_accuracy_all": hits / len(indices),
            "old_hits_all": int((old > 0).sum()),
            "selected_pairs": selected_count,
            "selected_hits": selected_hits,
            "selected_accuracy": selected_hits / selected_count if selected_count else None,
            "selected_coverage": selected_count / len(indices)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "features", "old_encoder", "new_encoder",
                 "calibration", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    if manifest["schema"] != "failure_rank_train_scene_v1" or \
            calibration["failure_rank_manifest_sha256"] != digest(args.manifest) or \
            calibration["old_encoder_sha256"] != digest(args.old_encoder) or \
            calibration["new_encoder_sha256"] != digest(args.new_encoder):
        raise ValueError("input provenance mismatch")
    records = manifest["pairs"]
    if len(records) != 928:
        raise ValueError("unexpected failure-pair coverage")
    indices = {split: [i for i, row in enumerate(records) if row["split"] == split]
               for split in ("fit", "development", "audit")}
    if [len(indices[x]) for x in ("fit", "development", "audit")] != \
            [697, 125, 106]:
        raise ValueError("unexpected scene split coverage")
    cache = torch.load(args.features, map_location="cpu", weights_only=True)
    if cache["manifest_sha256"] != digest(args.manifest) or \
            tuple(cache["hidden"].shape) != (1856, 4, 2048):
        raise ValueError("feature cache mismatch")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(928, 2, 4, 2048)
    scales = {"old": float(calibration["old_temporal_scale"]),
              "new": float(calibration["temporal_scale"])}
    if not all(math.isfinite(value) and 0 < value < 100 for value in scales.values()):
        raise ValueError("invalid fit-only score scale")
    deltas = {}
    with torch.inference_mode():
        for name, path in (("old", args.old_encoder), ("new", args.new_encoder)):
            model = load_encoder(path)
            # Small chunks cap CPU memory while the unrelated online run uses GPU.
            scored = []
            for first in range(0, len(records), 64):
                scored.append(predict(model, hidden[first:first + 64],
                                      torch.device("cpu"))[:, :, -1] / scales[name])
            value = torch.cat(scored)
            deltas[name] = value[:, 0] - value[:, 1]
            del model, scored, value
    scenes = [row["scene_id"] for row in records]
    table = {part: summarize(selected, deltas["old"], deltas["new"], scenes)
             for part, selected in indices.items()}
    def passes(part: str, minimum: int) -> bool:
        item = table[part]
        return item["selected_pairs"] >= minimum and \
            item["selected_accuracy"] >= .80 and \
            item["selected_accuracy"] >= item["new_accuracy_all"] + .05
    gate = {"rule": "Both held-out train-scene splits need >=80% selected-pair accuracy and >=5-point gain over the new encoder's ungated rate, with at least 60 development and 50 audit pairs.",
            "development_pass": passes("development", 60),
            "audit_pass": passes("audit", 50)}
    gate["pass"] = gate["development_pass"] and gate["audit_pass"]
    report = {"schema": "dual_frozen_encoder_agreement_train_screen_v1",
              "interpretation": "Exploratory training-scene representation screen after related audit data were inspected; no online or val-unseen navigation claim.",
              "manifest_sha256": digest(args.manifest),
              "feature_cache_sha256": digest(args.features),
              "old_encoder_sha256": digest(args.old_encoder),
              "new_encoder_sha256": digest(args.new_encoder),
              "calibration_sha256": digest(args.calibration),
              "group_size_in_source_rollouts": 4,
              "rule": "Apply a failed-pair rank only if frozen old and new terminal score differences have the same nonzero sign.",
              "table": table, "predeclared_online_pilot_gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"table": table, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
