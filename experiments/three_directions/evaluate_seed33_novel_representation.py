"""External episode-disjoint check of frozen temporal reward encoders.

The seed-11 and instruction-contrastive encoders are loaded without updates.
Seed-33 novel pairs and their same-scene swapped instructions were not used
for either encoder's training or epoch choice. Scenes may overlap training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, evaluate, predict
from train_temporal_contrastive_v2 import instruction_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--swaps-manifest", type=Path, required=True)
    parser.add_argument("--swaps-root", type=Path, required=True)
    parser.add_argument("--encoder-v1", type=Path, required=True)
    parser.add_argument("--encoder-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    swaps = json.loads(args.swaps_manifest.read_text())
    pairs = manifest["pairs"]
    if manifest["selection"]["group_size"] != 4 or \
            not manifest["selection"]["episode_disjoint_from_prior"] or \
            swaps["source_manifest_sha256"] != digest(args.manifest) or \
            len(swaps["pairs"]) != len(pairs):
        raise ValueError("novel set provenance mismatch")
    labels = json.loads((args.labels_root / "summary.json").read_text())
    wrong_summary = json.loads((args.swaps_root / "summary_audit.json").read_text())
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    if labels["manifest_sha256"] != digest(args.manifest) or \
            labels["completed_trajectories"] != 2 * len(pairs) or labels["errors"] or \
            wrong_summary["v2_manifest_sha256"] != digest(args.swaps_manifest) or \
            wrong_summary["records"] != len(pairs) or \
            cache["manifest_sha256"] != digest(args.manifest) or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048):
        raise ValueError("novel feature or label coverage mismatch")
    hidden = F.normalize(cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    stop = cache["stop_margin"].float().reshape(-1, 2, 4)
    true = torch.zeros(len(pairs), 2, 4)
    wrong = torch.zeros(len(pairs), 4, 2048)
    swap_rows = {row["pair_id"]: row for row in swaps["pairs"]}
    for index, pair in enumerate(pairs):
        if pair["pair_id"] not in swap_rows:
            raise ValueError("swap pair missing")
        for role_index, role in enumerate(("success", "failure")):
            record_id = pair["pair_id"] + "_" + role
            if cache["record_ids"][2 * index + role_index] != record_id:
                raise ValueError(f"feature order mismatch {record_id}")
            item = json.loads((args.labels_root / "records" /
                               f"{record_id}.json").read_text())
            if item["record_id"] != record_id:
                raise ValueError(f"distance label mismatch {record_id}")
            distances = item["distance_to_goal_m"]
            if len(distances) != 4 or distances[0] <= 0:
                raise ValueError(f"invalid distance {record_id}")
            true[index, role_index] = torch.tensor(
                [(distances[0] - distance) / distances[0]
                 for distance in distances])
        swapped = torch.load(args.swaps_root / "records" /
                             f"{pair['pair_id']}.pt", map_location="cpu",
                             weights_only=False)
        if swapped["pair_id"] != pair["pair_id"] or \
                swapped["manifest_sha256"] != digest(args.swaps_manifest) or \
                swapped["model_config_sha256"] != cache["model_config_sha256"] or \
                swapped["hidden"].shape != (4, 2048):
            raise ValueError(f"wrong-instruction state mismatch {pair['pair_id']}")
        wrong[index] = F.normalize(swapped["hidden"].float(), dim=-1)
    results = {}
    for name, path in (("temporal_v1", args.encoder_v1),
                       ("instruction_contrastive_v2", args.encoder_v2)):
        frozen = torch.load(path, map_location="cpu", weights_only=False)
        if frozen["feature_model_config_sha256"] != cache["model_config_sha256"]:
            raise ValueError(f"encoder/SFT mismatch {name}")
        model = TemporalPotential().cuda().eval()
        model.load_state_dict(frozen["model"])
        correct_pred = predict(model, hidden, torch.device("cuda:0"))
        wrong_pred = predict(model, wrong.unsqueeze(1), torch.device("cuda:0"))[:, 0]
        endpoint = evaluate(correct_pred, true, list(range(len(pairs))), pairs, stop)
        grounding = instruction_score(correct_pred, wrong_pred,
                                      list(range(len(pairs))), pairs)
        results[name] = {"encoder_sha256": digest(path),
                         "endpoint_and_temporal": endpoint,
                         "instruction_grounding": grounding,
                         "screen_conditions": {
                             "endpoint_at_least_0_75":
                                 endpoint["endpoint_success_over_failure"] >= .75,
                             "temporal_at_least_0_60":
                                 endpoint["temporal_concordance"] >= .60,
                             "instruction_at_least_0_75":
                                 grounding["correct_instruction_preference"] >= .75,
                             "endpoint_gain_over_raw_stop_at_least_5pp":
                                 endpoint["endpoint_success_over_failure"] -
                                 endpoint["raw_stop_reference"] >= .05}}
    report = {"schema": "seed33_novel_representation_check_v1",
              "interpretation": "Episode-disjoint train-split diagnostics; scenes can overlap; no RL or val-unseen result.",
              "pairs": len(pairs), "scenes": len(set(pair["scene_id"] for pair in pairs)),
              "manifest_sha256": digest(args.manifest),
              "swaps_manifest_sha256": digest(args.swaps_manifest),
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
