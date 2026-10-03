"""Train-scene screen for path-persistent temporal reward transforms.

The frozen encoder returns four causal potentials, with the initial value
exactly zero. This probe asks whether penalizing late backtracking improves
same-instruction failed-pair ranking without hurting success or grounding.
Only fit/development train scenes are opened. No audit or val-unseen label is
used to choose a transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


TRANSFORMS = {
    "terminal": lambda p: p[..., 3],
    "late_average": lambda p: .75 * p[..., 3] + .25 * p[..., 2],
    "backtrack_half": lambda p: p[..., 3] - .5 *
        (torch.maximum(p[..., 1], p[..., 2]) - p[..., 3]).clamp_min(0),
    "backtrack_full": lambda p: p[..., 3] -
        (torch.maximum(p[..., 1], p[..., 2]) - p[..., 3]).clamp_min(0),
}


def load_model(path: Path) -> TemporalPotential:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalPotential().eval()
    model.load_state_dict(state["model"])
    return model


def metric(margin: torch.Tensor, indices: list[int], scenes: list[str]) -> dict:
    value = margin[indices]
    return {"pairs": len(indices), "scenes": len({scenes[i] for i in indices}),
            "hits": int((value > 0).sum()), "rate": float((value > 0).float().mean()),
            "mean_margin": float(value.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("failure_manifest", "failure_features", "old_pair_manifest",
                 "old_v2_manifest", "old_features", "old_swaps", "old_encoder",
                 "new_encoder", "calibration", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    failure = json.loads(args.failure_manifest.read_text())
    old_pairs = json.loads(args.old_pair_manifest.read_text())["pairs"]
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    calibration = json.loads(args.calibration.read_text())
    if failure["schema"] != "failure_rank_train_scene_v1" or \
            old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            calibration["failure_rank_manifest_sha256"] != digest(args.failure_manifest) or \
            calibration["new_encoder_sha256"] != digest(args.new_encoder) or \
            calibration["old_encoder_sha256"] != digest(args.old_encoder):
        raise ValueError("input provenance mismatch")
    failed_pairs = failure["pairs"]
    split_old = {row["pair_id"]: row["split"] for row in old_v2["pairs"]}
    f_index = {name: [i for i, row in enumerate(failed_pairs)
                      if row["split"] == name] for name in ("fit", "development")}
    s_index = {name: [i for i, row in enumerate(old_pairs)
                      if split_old[row["pair_id"]] == name]
               for name in ("fit", "development")}
    if [len(f_index[x]) for x in ("fit", "development")] != [697, 125] or \
            [len(s_index[x]) for x in ("fit", "development")] != [300, 52]:
        raise ValueError("unexpected fit/development coverage")
    f_cache = torch.load(args.failure_features, map_location="cpu", weights_only=False)
    s_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    if f_cache["manifest_sha256"] != digest(args.failure_manifest) or \
            s_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            f_cache["hidden"].shape != (1856, 4, 2048) or \
            s_cache["hidden"].shape != (800, 4, 2048):
        raise ValueError("feature cache mismatch")
    f_hidden = F.normalize(f_cache["hidden"].float(), dim=-1).reshape(928, 2, 4, 2048)
    s_hidden = F.normalize(s_cache["hidden"].float(), dim=-1).reshape(400, 2, 4, 2048)
    wrong = torch.zeros(400, 4, 2048)
    for split in ("fit", "development"):
        for i in s_index[split]:
            pair_id = old_pairs[i]["pair_id"]
            row = torch.load(args.old_swaps / "records" / f"{pair_id}.pt",
                             map_location="cpu", weights_only=False)
            if row["pair_id"] != pair_id or row["hidden"].shape != (4, 2048):
                raise ValueError("wrong-instruction feature mismatch")
            wrong[i] = F.normalize(row["hidden"].float(), dim=-1)
    active_f = f_index["fit"] + f_index["development"]
    active_s = s_index["fit"] + s_index["development"]
    f_scenes = [row["scene_id"] for row in failed_pairs]
    s_scenes = [row["scene_id"] for row in old_pairs]
    table = {}
    for encoder_name, path in (("old", args.old_encoder), ("new", args.new_encoder)):
        model = load_model(path)
        f_pred = predict(model, f_hidden[active_f], torch.device("cpu"))
        s_pred = predict(model, s_hidden[active_s], torch.device("cpu"))
        w_pred = predict(model, wrong[active_s].unsqueeze(1), torch.device("cpu"))[:, 0]
        for transform_name, transform in TRANSFORMS.items():
            f = transform(f_pred)
            s = transform(s_pred)
            w = transform(w_pred)
            f_margin = torch.zeros(928)
            s_margin = torch.zeros(400)
            g_margin = torch.zeros(400)
            f_margin[active_f] = f[:, 0] - f[:, 1]
            s_margin[active_s] = s[:, 0] - s[:, 1]
            g_margin[active_s] = s[:, 0] - w
            table[f"{encoder_name}_{transform_name}"] = {
                split: {
                    "failure_near_over_far": metric(f_margin, f_index[split], f_scenes),
                    "success_over_failure": metric(s_margin, s_index[split], s_scenes),
                    "correct_over_wrong_instruction": metric(g_margin, s_index[split], s_scenes)}
                for split in ("fit", "development")}
    new_f = table["new_terminal"]["development"]["failure_near_over_far"]["hits"]
    old_s = table["old_terminal"]["development"]["success_over_failure"]["hits"]
    old_g = table["old_terminal"]["development"]["correct_over_wrong_instruction"]["hits"]
    for key, row in table.items():
        dev = row["development"]
        row["development_gate"] = {
            "failure_at_least_3pp_above_new_terminal":
                dev["failure_near_over_far"]["hits"] >= new_f + 4,
            "success_drop_at_most_2pp_from_old_terminal":
                dev["success_over_failure"]["hits"] >= old_s - 1,
            "grounding_drop_at_most_2pp_from_old_terminal":
                dev["correct_over_wrong_instruction"]["hits"] >= old_g - 1}
    eligible = [key for key in table if not key.endswith("terminal")
                and all(table[key]["development_gate"].values())]
    result = {"schema": "temporal_persistence_train_scene_screen_v1",
              "interpretation": "Exploratory train-scene fit/development screen only. Prior probes have inspected related development data; no audit or held-out navigation claim.",
              "failure_manifest_sha256": digest(args.failure_manifest),
              "old_v2_manifest_sha256": digest(args.old_v2_manifest),
              "old_encoder_sha256": digest(args.old_encoder),
              "new_encoder_sha256": digest(args.new_encoder),
              "calibration_sha256": digest(args.calibration),
              "formulas": {
                  "late_average": "0.75*p3+0.25*p2",
                  "backtrack_half": "p3-0.5*relu(max(p1,p2)-p3)",
                  "backtrack_full": "p3-relu(max(p1,p2)-p3)"},
              "selection_rule": ">=4 extra failed-pair hits of 125 over new terminal; <=1 lost hit of 52 on both success and grounding versus old terminal",
              "table": table, "eligible_development_only": eligible}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"eligible_development_only": eligible,
                      "development": {name: row["development"] for name, row in table.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
