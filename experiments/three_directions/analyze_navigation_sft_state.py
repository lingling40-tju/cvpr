"""Scene-disjoint train-only audit of navigation-SFT STOP and hidden features.

The STOP first-token margin is frozen. A regularized linear pairwise probe
uses only fit scenes, selects its L2 coefficient on development scenes, and
reads audit scenes once. This is not a navigation policy evaluation.
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


def scene_interval(hits: torch.Tensor, scenes: list[str], seed: int = 20261003) -> list[float]:
    unique = sorted(set(scenes))
    by_scene = {scene: [i for i, s in enumerate(scenes) if s == scene]
                for scene in unique}
    rng = random.Random(seed)
    draws = []
    for _ in range(5000):
        selected = [index for _ in unique
                    for index in by_scene[rng.choice(unique)]]
        draws.append(hits[selected].float().mean().item())
    draws.sort()
    return [draws[int(.025 * len(draws))], draws[int(.975 * len(draws))]]


def evaluate(values: torch.Tensor, scenes: list[str]) -> dict:
    if len(values) != len(scenes) or not torch.isfinite(values).all():
        raise ValueError("invalid paired score vector")
    hits = values > 0
    return {"pairs": len(values), "scenes": len(set(scenes)),
            "success_over_failure_accuracy": hits.float().mean().item(),
            "margin_mean": values.mean().item(),
            "scene_bootstrap95": scene_interval(hits, scenes),
            "per_scene": [{"scene": scene, "pairs": sum(s == scene for s in scenes),
                           "accuracy": hits[[i for i, s in enumerate(scenes)
                                             if s == scene]].float().mean().item()}
                          for scene in sorted(set(scenes))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    pairs = manifest["pairs"]
    if cache["schema"] != "navigation_sft_state_v1" or \
            cache["manifest_sha256"] != digest(args.manifest) or \
            cache["hidden"].shape != (2 * len(pairs), 4, 2048) or \
            cache["stop_margin"].shape != (2 * len(pairs), 4):
        raise ValueError("SFT feature coverage mismatch")
    for i, pair in enumerate(pairs):
        if cache["record_ids"][2 * i:2 * i + 2] != [
                pair["pair_id"] + "_success", pair["pair_id"] + "_failure"]:
            raise ValueError("SFT record order mismatch")
    hidden = F.normalize(cache["hidden"].float(), dim=-1)
    sft_margin = cache["stop_margin"].float()
    success = hidden[0::2]
    failure = hidden[1::2]
    deltas = success[:, -1] - failure[:, -1]
    split_ids = {split: [i for i, pair in enumerate(pairs)
                         if pair["split"] == split]
                 for split in ("fit", "development", "audit")}
    if {key: len(value) for key, value in split_ids.items()} != manifest["counts"]:
        raise ValueError("scene split count mismatch")
    scene_ids = {split: [pairs[i]["scene_id"] for i in ids]
                 for split, ids in split_ids.items()}
    raw = {}
    for split, ids in split_ids.items():
        success_readiness = sft_margin[0::2][ids]
        failure_readiness = sft_margin[1::2][ids]
        margin = success_readiness - failure_readiness
        raw[split] = {
            "last_frame": evaluate(margin[:, -1], scene_ids[split]),
            "last_two_max": evaluate(success_readiness[:, -2:].max(-1).values -
                                     failure_readiness[:, -2:].max(-1).values,
                                     scene_ids[split]),
            "success_endpoint_above_start_rate":
                (sft_margin[0::2][ids, -1] > sft_margin[0::2][ids, 0]).float().mean().item(),
            "failure_endpoint_above_start_rate":
                (sft_margin[1::2][ids, -1] > sft_margin[1::2][ids, 0]).float().mean().item(),
        }
    fit = deltas[split_ids["fit"]]
    development = deltas[split_ids["development"]]
    gram = fit @ fit.T
    identity = torch.eye(len(fit), dtype=gram.dtype)
    ones = torch.ones(len(fit), dtype=gram.dtype)
    candidates = []
    for regularization in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        weight = fit.T @ torch.linalg.solve(gram + regularization * identity, ones)
        dev_margin = development @ weight
        candidates.append(((dev_margin > 0).float().mean().item(),
                           dev_margin.mean().item(), regularization, weight))
    chosen = max(candidates, key=lambda row: (row[0], row[1], -row[2]))
    _, _, regularization, weight = chosen
    probe = {"regularization": regularization,
             "fit": evaluate(fit @ weight, scene_ids["fit"]),
             "development": evaluate(development @ weight,
                                     scene_ids["development"]),
             "audit": evaluate(deltas[split_ids["audit"]] @ weight,
                               scene_ids["audit"]),
             "all_development_candidates": [
                 {"regularization": row[2],
                  "accuracy": row[0]}
                 for row in candidates],
             "audit_success_endpoint_above_start_rate":
                 (((success[:, -1] - success[:, 0]) @ weight)[split_ids["audit"]] > 0).float().mean().item(),
             "audit_failure_endpoint_above_start_rate":
                 (((failure[:, -1] - failure[:, 0]) @ weight)[split_ids["audit"]] > 0).float().mean().item()}
    report = {"interpretation": "Frozen navigation-SFT train-scene representation audit; no RL or val-unseen result.",
              "model_name": cache["model_name"],
              "manifest_sha256": cache["manifest_sha256"],
              "raw_stop_readiness": raw,
              "regularized_hidden_probe": probe}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.probe_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight, "regularization": regularization,
                "model_config_sha256": cache["model_config_sha256"],
                "manifest_sha256": cache["manifest_sha256"]}, args.probe_output)
    print(json.dumps({"raw_stop_audit": raw["audit"],
                      "probe_audit": probe["audit"],
                      "probe_regularization": regularization}, indent=2))


if __name__ == "__main__":
    main()
