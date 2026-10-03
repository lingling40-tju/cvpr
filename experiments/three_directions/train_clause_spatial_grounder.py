"""Fit one frozen-protocol spatial clause grounder on R2R train scenes.

All visual/text towers are frozen. No val-unseen episode or navigation
metric is used for fitting or checkpoint choice.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_clause_alignment_manifest import digest


def load_part(part: str, args: argparse.Namespace, manifest: dict,
              ordinal: dict, device: torch.device) -> dict:
    rows = manifest["selected"][part]
    text = torch.load(args.clause_root / f"{part}_text.pt", map_location="cpu",
                      weights_only=True)
    visual = torch.load(args.ordinal_root / f"siglip_full_{part}.pt",
                        map_location="cpu", weights_only=True)
    if text["episode_ids"].tolist() != [r["episode_id"] for r in rows] or \
            visual["episode_ids"].tolist() != [r["episode_id"] for r in rows] or \
            text["model_config_sha256"] != visual["model_config_sha256"] or \
            text["manifest_sha256"] != digest(args.manifest):
        raise ValueError(f"text/visual identity mismatch: {part}")
    by_id = {row["episode_id"]: i for i, row in enumerate(rows)}
    frames = defaultdict(list)
    for i, (eid, offset) in enumerate(zip(visual["frame_episode_ids"].tolist(),
                                          visual["frame_offsets"].tolist())):
        frames[eid].append((offset, i))
    images = []
    patches = []
    for row in rows:
        eid = row["episode_id"]
        indices = [i for _, i in sorted(frames[eid])]
        if len(indices) != 6 or [visual["frame_offsets"][i].item()
                                 for i in indices] != list(range(6)):
            raise ValueError(f"frame coverage mismatch: {part}/{eid}")
        images.append(visual["images"][indices])
        payload = torch.load(args.clause_root / "spatial_tokens" / part /
                             "records" / f"{eid}.pt", map_location="cpu",
                             weights_only=True)
        if payload["record_sha256"] != row["record_sha256"] or \
                payload["manifest_sha256"] != digest(args.manifest) or \
                payload["config_sha256"] != text["model_config_sha256"]:
            raise ValueError(f"spatial feature identity mismatch: {part}/{eid}")
        patches.append(payload["patches"])
    offsets = text["offsets"].tolist()
    clauses = [F.normalize(text["clauses"][a:b].float().to(device), dim=-1)
               for a, b in zip(offsets[:-1], offsets[1:])]
    image_tensor = F.normalize(torch.stack(images).float().to(device), dim=-1)
    patch_tensor = F.normalize(torch.stack(patches).float().to(device), dim=-1)
    full = F.normalize(text["full"].float().to(device), dim=-1)
    pairs = []
    for pair in ordinal["subsets"][part]["pairs"]:
        a, b = by_id[pair["left"]], by_id[pair["right"]]
        if rows[a]["scene_id"] != pair["scene"] or rows[b]["scene_id"] != pair["scene"]:
            raise ValueError("pair scene mismatch")
        pairs.append((a, b, pair["scene"]))
    if len(pairs) != {"fit": 128, "calibration": 32}[part]:
        raise ValueError("natural pair count changed")
    return {"rows": rows, "images": image_tensor, "patches": patch_tensor,
            "full": full, "clauses": clauses, "pairs": pairs}


class SpatialGrounder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(768, 8) * 0.02)
        self.b = nn.Parameter(torch.zeros(8, 768))
        self.coeff_logit = nn.Parameter(torch.tensor(math.atanh(0.25)))

    def path(self, patches: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        similarity = torch.einsum("tpd,kd->tpk", patches, query).max(dim=1).values
        dp = similarity.new_full((query.shape[0],), -1e6)
        dp = torch.cat((similarity[:1, :1].flatten(), dp[1:]))
        for t in range(1, 6):
            dp = similarity[t] + torch.cummax(dp, dim=0).values
        return dp[-1] / 6

    def spatial(self, patches: torch.Tensor, clauses: torch.Tensor) -> torch.Tensor:
        query = F.normalize(clauses + clauses @ self.a @ self.b, dim=-1)
        return self.path(patches, query) - self.path(patches[:1].expand_as(patches), query)


def global_score(part: dict, route: int, instruction: int) -> torch.Tensor:
    return part["images"][route, -2:].mean(dim=0) @ part["full"][instruction]


def pair_scores(model: SpatialGrounder, part: dict, pair: tuple[int, int, str],
                gscale: float, sscale: float) -> tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor, torch.Tensor]:
    a, b, _ = pair
    scores = []
    for route, instruction in ((a, a), (a, b), (b, b), (b, a)):
        g = global_score(part, route, instruction) / gscale
        spatial = model.spatial(part["patches"][route],
                                part["clauses"][instruction]) / sscale
        scores.append(g + torch.tanh(model.coeff_logit) * spatial)
    return scores[0] - scores[1], scores[2] - scores[3], \
           global_score(part, a, a) - global_score(part, a, b), \
           global_score(part, b, b) - global_score(part, b, a)


def fit_scales(model: SpatialGrounder, part: dict) -> tuple[float, float]:
    global_margins = []
    spatial_margins = []
    with torch.no_grad():
        for a, b, _ in part["pairs"]:
            global_margins.extend((global_score(part, a, a) - global_score(part, a, b),
                                   global_score(part, b, b) - global_score(part, b, a)))
            spatial_margins.extend((model.spatial(part["patches"][a], part["clauses"][a]) -
                                    model.spatial(part["patches"][a], part["clauses"][b]),
                                    model.spatial(part["patches"][b], part["clauses"][b]) -
                                    model.spatial(part["patches"][b], part["clauses"][a])))
    gs = torch.stack(global_margins).std(unbiased=False).item()
    ss = torch.stack(spatial_margins).std(unbiased=False).item()
    if not 1e-6 < gs < 1 or not 1e-6 < ss < 1:
        raise ValueError(f"degenerate fit-only scales: {gs}, {ss}")
    return gs, ss


def evaluate(model: SpatialGrounder, part: dict, gscale: float,
             sscale: float) -> dict:
    rows = []
    by_scene = defaultdict(lambda: {"model": [], "baseline": []})
    with torch.no_grad():
        for a, b, scene in part["pairs"]:
            ma, mb, ga, gb = pair_scores(model, part, (a, b, scene), gscale, sscale)
            flags = {"model": [int(ma.item() > 0), int(mb.item() > 0)],
                     "baseline": [int(ga.item() > 0), int(gb.item() > 0)]}
            for key in flags:
                by_scene[scene][key].extend(flags[key])
            rows.append({"episode_ids": [part["rows"][a]["episode_id"],
                                         part["rows"][b]["episode_id"]],
                         "scene_id": scene, "model_margins": [ma.item(), mb.item()],
                         "baseline_margins": [ga.item(), gb.item()],
                         "correct": flags})
    metrics = {}
    for key in ("model", "baseline"):
        flags = [v for row in rows for v in row["correct"][key]]
        metrics[key] = {"correct": sum(flags), "comparisons": len(flags),
                        "accuracy": sum(flags) / len(flags),
                        "strict_pairs_correct": sum(all(row["correct"][key]) for row in rows),
                        "pairs": len(rows),
                        "scene_macro_accuracy": sum(sum(v[key]) / len(v[key]) for v in by_scene.values()) /
                                                len(by_scene)}
    scenes = list(by_scene.values())
    rng = random.Random(11)
    differences = []
    for _ in range(2000):
        sample = [rng.choice(scenes) for _ in scenes]
        difference = (sum(sum(v["model"]) - sum(v["baseline"]) for v in sample) /
                      sum(len(v["model"]) for v in sample))
        differences.append(difference)
    differences.sort()
    return {"metrics": metrics, "paired_scene_bootstrap_difference_95":
            [differences[50], differences[1949]], "pairs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--ordinal-root", type=Path, required=True)
    parser.add_argument("--clause-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.cache_audit.read_text())
    if audit["schema"] != "clause_spatial_cache_audit_v1" or \
            audit["source_hashes"]["manifest_sha256"] != digest(args.manifest) or \
            not audit["scene_disjoint"]:
        raise ValueError("unverified source cache")
    manifest = json.loads(args.manifest.read_text())
    ordinal = json.loads(args.ordinal_manifest.read_text())
    if manifest["ordinal_manifest_sha256"] != digest(args.ordinal_manifest):
        raise ValueError("ordinal manifest changed")
    torch.manual_seed(11)
    device = torch.device("cuda")
    fit = load_part("fit", args, manifest, ordinal, device)
    calibration = load_part("calibration", args, manifest, ordinal, device)
    model = SpatialGrounder().to(device)
    gscale, sscale = fit_scales(model, fit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rng = random.Random(11)
    epoch_losses = []
    for epoch in range(64):
        order = list(range(len(fit["pairs"])))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), 16):
            optimizer.zero_grad(set_to_none=True)
            margins = []
            for i in order[start:start + 16]:
                ma, mb, _, _ = pair_scores(model, fit, fit["pairs"][i], gscale, sscale)
                margins.extend((ma, mb))
            loss = F.softplus(-torch.stack(margins)).mean()
            loss = loss + 0.01 * (model.a.square().mean() + model.b.square().mean())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(sum(losses) / len(losses))
        if (epoch + 1) % 8 == 0:
            print(f"epoch={epoch + 1}/64 loss={epoch_losses[-1]:.5f} "
                  f"coefficient={torch.tanh(model.coeff_logit).item():.4f}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "grounder_epoch64.pt"
    torch.save({"schema": "clause_spatial_grounder_v1", "seed": 11,
                "epoch": 64, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "fit_global_margin_std": gscale, "fit_spatial_margin_std": sscale,
                "manifest_sha256": digest(args.manifest),
                "cache_audit_sha256": digest(args.cache_audit)}, checkpoint)
    fit_result = evaluate(model, fit, gscale, sscale)
    calibration_result = evaluate(model, calibration, gscale, sscale)
    candidate = calibration_result["metrics"]["model"]
    baseline = calibration_result["metrics"]["baseline"]
    passed = (candidate["correct"] >= 52 and
              candidate["scene_macro_accuracy"] > baseline["scene_macro_accuracy"])
    report = {"schema": "clause_spatial_grounder_screen_v1",
              "interpretation": "exploratory train-scene offline instruction grounding, no navigation result",
              "fixed_protocol": {"seed": 11, "epochs": 64, "batch_pairs": 16,
                                 "rank": 8, "adamw_lr": 1e-3, "weight_decay": 0.01,
                                 "calibration_gate": "at least 52/64 and scene macro above frozen whole instruction"},
              "source_hashes": {"manifest": digest(args.manifest),
                                "ordinal_manifest": digest(args.ordinal_manifest),
                                "cache_audit": digest(args.cache_audit),
                                "checkpoint": digest(checkpoint)},
              "fit_only_scales": {"global": gscale, "spatial": sscale},
              "final_coefficient": torch.tanh(model.coeff_logit).item(),
              "epoch_losses": epoch_losses,
              "fit": fit_result, "calibration": calibration_result,
              "calibration_gate_passed": passed}
    path = args.output_root / "screen.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint_sha256": digest(checkpoint),
                      "fit_metrics": fit_result["metrics"],
                      "calibration_metrics": calibration_result["metrics"],
                      "calibration_gate_passed": passed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
