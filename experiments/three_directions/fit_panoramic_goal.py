"""Fit goal-panorama/text alignment; audit transfer to ordinary expert views.

Uses R2R train scenes only. The four goal-pose views and train goal positions
are supervision preparation. The candidate online score receives only a
current RGB embedding and instruction embedding.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_goal_contrastive import GoalMatcher
from fit_temporal_counterfactual import load_sequences


def load_views(path: Path, manifest: dict, subset: str, device: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    ids = manifest["subsets"][subset]["episode_ids"]
    if data["subset"] != subset or data["episode_ids"].tolist() != ids:
        raise ValueError("panorama manifest/feature coverage mismatch")
    if data["images"].shape[:2] != (len(ids), 4) or \
            data["texts"].shape[0] != len(ids) or \
            len(data["scene_ids"]) != len(ids):
        raise ValueError("invalid panorama feature shape")
    if not torch.isfinite(data["images"]).all() or not torch.isfinite(data["texts"]).all():
        raise ValueError("nonfinite panorama features")
    index = {eid: i for i, eid in enumerate(ids)}
    pairs = []
    for pair in manifest["subsets"][subset]["pairs"]:
        a, b = index[pair["left"]], index[pair["right"]]
        if data["scene_ids"][a] != pair["scene"] or \
                data["scene_ids"][b] != pair["scene"]:
            raise ValueError("pair scene mismatch")
        pairs.append((a, b, pair["scene"]))
    return {"ids": ids, "scenes": data["scene_ids"],
            "images": data["images"].to(device), "texts": data["texts"].to(device),
            "pairs": pairs, "backbone": data["backbone"],
            "model_config_sha256": data["model_config_sha256"]}


def subset(data: dict, scenes: set[str]) -> dict:
    selected = [i for i, scene in enumerate(data["scenes"]) if scene in scenes]
    if {data["scenes"][i] for i in selected} != scenes:
        raise ValueError("scene subset incomplete")
    remap = {old: new for new, old in enumerate(selected)}
    pairs = [(remap[a], remap[b], scene) for a, b, scene in data["pairs"]
             if a in remap and b in remap]
    if not pairs:
        raise ValueError("empty scene subset pairs")
    return {"ids": [data["ids"][i] for i in selected],
            "scenes": [data["scenes"][i] for i in selected],
            "images": data["images"][selected], "texts": data["texts"][selected],
            "pairs": pairs}


def score(model: GoalMatcher, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
    return model(images, texts[:, None, :]).max(dim=1).values


@torch.no_grad()
def evaluate_panoramas(model: GoalMatcher, data: dict) -> dict:
    model.eval()
    pairs = torch.tensor([(a, b) for a, b, _ in data["pairs"]],
                         dtype=torch.long, device=data["images"].device)
    a, b = pairs.T
    ia, ib, ta, tb = data["images"][a], data["images"][b], \
        data["texts"][a], data["texts"][b]
    aa, ab, ba, bb = score(model, ia, ta), score(model, ia, tb), \
        score(model, ib, ta), score(model, ib, tb)
    raw = lambda x, t: (x * t[:, None, :]).sum(-1).max(-1).values
    ra, rb, rc, rd = raw(ia, ta), raw(ia, tb), raw(ib, ta), raw(ib, tb)
    return {"episodes": len(data["ids"]), "scenes": len(set(data["scenes"])),
            "pairs": len(pairs), "comparisons": 2 * len(pairs),
            "image_text_accuracy": torch.cat(((aa > ab), (bb > ba))).float().mean().item(),
            "text_image_accuracy": torch.cat(((aa > ba), (bb > ab))).float().mean().item(),
            "image_text_margin_mean": torch.cat((aa - ab, bb - ba)).mean().item(),
            "raw_image_text_accuracy":
                torch.cat(((ra > rb), (rd > rc))).float().mean().item(),
            "raw_text_image_accuracy":
                torch.cat(((ra > rc), (rd > rb))).float().mean().item()}


@torch.no_grad()
def evaluate_expert_transfer(model: GoalMatcher, data: dict, scenes: set[str]) -> dict:
    model.eval()
    indices = [i for i, scene in enumerate(data["scenes"]) if scene in scenes]
    if {data["scenes"][i] for i in indices} != scenes:
        raise ValueError("expert transfer scenes incomplete")
    remap = {old: new for new, old in enumerate(indices)}
    pairs = [(remap[a], remap[b]) for a, b in data["pairs"]
             if a in remap and b in remap]
    if not pairs:
        raise ValueError("expert transfer pairs missing")
    images, texts = data["images"][indices], data["texts"][indices]
    pair_tensor = torch.tensor(pairs, dtype=torch.long, device=images.device)
    a, b = pair_tensor.T
    aa = score(model, images[a, -2:], texts[a])
    ab = score(model, images[a, -2:], texts[b])
    bb = score(model, images[b, -2:], texts[b])
    ba = score(model, images[b, -2:], texts[a])
    end = score(model, images[:, -2:], texts)
    start = score(model, images[:, :1], texts)
    raw = lambda x, t: (x * t[:, None, :]).sum(-1).max(-1).values
    return {"episodes": len(indices), "scenes": len(scenes), "pairs": len(pairs),
            "comparisons": 2 * len(pairs),
            "image_text_accuracy": torch.cat(((aa > ab), (bb > ba))).float().mean().item(),
            "endpoint_above_start_rate": (end > start).float().mean().item(),
            "raw_image_text_accuracy": torch.cat((
                raw(images[a, -2:], texts[a]) > raw(images[a, -2:], texts[b]),
                raw(images[b, -2:], texts[b]) > raw(images[b, -2:], texts[a]))).float().mean().item(),
            "raw_endpoint_above_start_rate":
                (raw(images[:, -2:], texts) > raw(images[:, :1], texts)).float().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--expert-calibration-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(args.manifest.read_text())
    ordinal = json.loads(args.ordinal_manifest.read_text())
    if manifest["fit_scenes"] != ordinal["fit_scenes"] or \
            manifest["calibration_scenes"] != ordinal["calibration_scenes"]:
        raise ValueError("scene split changed")
    fit = load_views(args.fit_features, manifest, "fit", device)
    calibration = load_views(args.calibration_features, manifest, "calibration", device)
    expert = load_sequences(args.expert_calibration_features, ordinal,
                            "calibration", device)
    if fit["backbone"] != calibration["backbone"] or \
            fit["model_config_sha256"] != calibration["model_config_sha256"] or \
            fit["model_config_sha256"] != expert["model_config_sha256"]:
        raise ValueError("frozen backbone mismatch")
    scenes = manifest["calibration_scenes"]
    development = subset(calibration, set(scenes[:5]))
    audit = subset(calibration, set(scenes[5:]))
    model = GoalMatcher(fit["images"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    by_scene = defaultdict(list)
    for a, b, scene in fit["pairs"]:
        by_scene[scene].append((a, b))
    active_scenes = sorted(by_scene)
    best, best_step, best_state, best_development = -float("inf"), 0, None, None
    for step in range(1, args.steps + 1):
        model.train()
        selected_pairs = [rng.choice(by_scene[rng.choice(active_scenes)]) for _ in range(64)]
        pairs = torch.tensor(selected_pairs, dtype=torch.long, device=device)
        a, b = pairs.T
        ia, ib, ta, tb = fit["images"][a], fit["images"][b], \
            fit["texts"][a], fit["texts"][b]
        aa, ab, ba, bb = score(model, ia, ta), score(model, ia, tb), \
            score(model, ib, ta), score(model, ib, tb)
        # Both image->text and text->image preferences use natural same-start
        # pairs; no goal coordinate enters the model's input.
        margins = torch.cat((aa - ab, bb - ba, aa - ba, bb - ab))
        loss = F.softplus(0.3 - margins).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            dev = evaluate_panoramas(model, development)
            criterion = dev["image_text_accuracy"] + dev["text_image_accuracy"]
            if criterion > best:
                best, best_step = criterion, step
                best_state = copy.deepcopy({key: value.detach().cpu()
                                            for key, value in model.state_dict().items()})
                best_development = dev
            if step - best_step >= 200:
                break
    if best_state is None:
        raise RuntimeError("no development checkpoint selected")
    model.load_state_dict(best_state)
    goal_audit = evaluate_panoramas(model, audit)
    transfer = evaluate_expert_transfer(model, expert, set(scenes[5:]))
    gate = {"goal_image_text_at_least_0_75": goal_audit["image_text_accuracy"] >= .75,
            "goal_text_image_at_least_0_75": goal_audit["text_image_accuracy"] >= .75,
            "expert_image_text_at_least_0_75": transfer["image_text_accuracy"] >= .75,
            "expert_cf_above_raw_by_0_05": transfer["image_text_accuracy"] -
                transfer["raw_image_text_accuracy"] >= .05,
            "expert_endpoint_above_start_at_least_0_65":
                transfer["endpoint_above_start_rate"] >= .65}
    gate["status"] = "passed" if all(gate.values()) else "failed"
    report = {"interpretation": "Train-scene panorama alignment and expert-view transfer; no RL result.",
              "seed": args.seed, "best_step": best_step,
              "fit_episodes": len(fit["ids"]), "fit_pairs": len(fit["pairs"]),
              "development_scenes": scenes[:5], "audit_scenes": scenes[5:],
              "development": best_development, "goal_panorama_audit": goal_audit,
              "expert_view_transfer_audit": transfer, "candidate_gate": gate}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": best_state, "seed": args.seed,
                "feature_dim": fit["images"].shape[-1], "best_step": best_step},
               args.output_dir / "panoramic_goal_head.pt")
    print(json.dumps({"goal_panorama_audit": goal_audit,
                      "expert_view_transfer_audit": transfer,
                      "candidate_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
