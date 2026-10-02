"""Exploratory goal-image/instruction matching on frozen train-only SigLIP features.

The only supervised comparisons are natural R2R same-start, different-goal
instruction pairs. This screen does not train an RL policy or use val-unseen.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalMatcher(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.image = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, dim))
        self.text = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, dim))
        self.log_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        visual = F.normalize(images + 0.1 * self.image(images), dim=-1)
        language = F.normalize(texts + 0.1 * self.text(texts), dim=-1)
        return self.log_scale.exp().clamp(max=100) * (visual * language).sum(-1)


def load(path: Path, manifest: dict, subset: str, device: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    ids = data["episode_ids"].tolist()
    if data["subset"] != subset or ids != manifest["subsets"][subset]["episode_ids"]:
        raise ValueError("feature subset and coverage mismatch")
    if len(data["scene_ids"]) != len(ids):
        raise ValueError("scene coverage mismatch")
    frame_map = defaultdict(list)
    for index, eid in enumerate(data["frame_episode_ids"].tolist()):
        frame_map[eid].append(index)
    images = []
    for eid in ids:
        frames = sorted(frame_map[eid], key=lambda index: data["frame_offsets"][index].item())
        if len(frames) != 6:
            raise ValueError(f"episode {eid}: expected six expert frames")
        images.append(data["images"][frames])
    pairs = []
    index = {eid: i for i, eid in enumerate(ids)}
    for row in manifest["subsets"][subset]["pairs"]:
        a, b = row["left"], row["right"]
        if a not in index or b not in index:
            raise ValueError("missing natural pair")
        if data["scene_ids"][index[a]] != data["scene_ids"][index[b]]:
            raise ValueError("pair scene mismatch")
        pairs.append((index[a], index[b]))
    return {"ids": ids, "scenes": data["scene_ids"],
            "images": torch.stack(images).to(device), "texts": data["texts"].to(device),
            "pairs": pairs, "backbone": data["backbone"],
            "model_config_sha256": data["model_config_sha256"]}


def scene_subset(data: dict, allowed: set[str]) -> dict:
    indices = [i for i, scene in enumerate(data["scenes"]) if scene in allowed]
    if {data["scenes"][i] for i in indices} != allowed:
        raise ValueError("missing held-out scene")
    remap = {old: new for new, old in enumerate(indices)}
    pairs = [(remap[a], remap[b]) for a, b in data["pairs"] if a in remap and b in remap]
    if not pairs:
        raise ValueError("empty held-out pairs")
    return {"ids": [data["ids"][i] for i in indices],
            "scenes": [data["scenes"][i] for i in indices],
            "images": data["images"][indices], "texts": data["texts"][indices],
            "pairs": pairs}


@torch.no_grad()
def evaluate(model: GoalMatcher, data: dict) -> dict:
    model.eval()
    pairs = torch.tensor(data["pairs"], dtype=torch.long, device=data["images"].device)
    a, b = pairs.T
    frames_a, frames_b = data["images"][a, -2:], data["images"][b, -2:]
    text_a, text_b = data["texts"][a], data["texts"][b]
    correct = torch.cat((model(frames_a, text_a[:, None, :]).flatten(),
                         model(frames_b, text_b[:, None, :]).flatten()))
    wrong = torch.cat((model(frames_a, text_b[:, None, :]).flatten(),
                       model(frames_b, text_a[:, None, :]).flatten()))
    raw_correct = torch.cat(((frames_a * text_a[:, None, :]).sum(-1).flatten(),
                             (frames_b * text_b[:, None, :]).sum(-1).flatten()))
    raw_wrong = torch.cat(((frames_a * text_b[:, None, :]).sum(-1).flatten(),
                           (frames_b * text_a[:, None, :]).sum(-1).flatten()))
    endpoint = model(data["images"][:, -1], data["texts"])
    start = model(data["images"][:, 0], data["texts"])
    return {"episodes": len(data["ids"]), "scenes": len(set(data["scenes"])),
            "pairs": len(pairs), "comparisons": len(correct),
            "counterfactual_accuracy": (correct > wrong).float().mean().item(),
            "counterfactual_margin_mean": (correct - wrong).mean().item(),
            "raw_backbone_counterfactual_accuracy":
                (raw_correct > raw_wrong).float().mean().item(),
            "endpoint_above_start_rate": (endpoint > start).float().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(args.manifest.read_text())
    fit = load(args.fit_features, manifest, "fit", device)
    calibration = load(args.calibration_features, manifest, "calibration", device)
    if fit["backbone"] != calibration["backbone"] or \
            fit["model_config_sha256"] != calibration["model_config_sha256"]:
        raise ValueError("backbone mismatch")
    scenes = manifest["calibration_scenes"]
    development = scene_subset(calibration, set(scenes[:5]))
    audit = scene_subset(calibration, set(scenes[5:]))
    pairs = torch.tensor(fit["pairs"], dtype=torch.long, device=device)
    model = GoalMatcher(fit["images"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    best_score, best_step, best_state, best_development = -float("inf"), 0, None, None
    for step in range(1, args.steps + 1):
        model.train()
        choice = torch.randint(len(pairs), (32,), generator=generator, device=device)
        a, b = pairs[choice].T
        # Late expert frames show each destination. The two texts share a
        # physical start but name different goals, so an image-only shortcut
        # cannot satisfy both inequalities.
        frame_choice = torch.randint(4, 6, (len(a),), generator=generator, device=device)
        ia, ib = fit["images"][a, frame_choice], fit["images"][b, frame_choice]
        ta, tb = fit["texts"][a], fit["texts"][b]
        aa, ab = model(ia, ta), model(ia, tb)
        bb, ba = model(ib, tb), model(ib, ta)
        loss = F.softplus(0.4 - (aa - ab)).mean() + F.softplus(0.4 - (bb - ba)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            metrics = evaluate(model, development)
            score = metrics["counterfactual_accuracy"] + 0.001 * metrics["counterfactual_margin_mean"]
            if score > best_score:
                best_score, best_step = score, step
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                best_development = metrics
            if step - best_step >= 150:
                break
    if best_state is None:
        raise RuntimeError("no selected checkpoint")
    model.load_state_dict(best_state)
    audited = evaluate(model, audit)
    report = {"interpretation": "Exploratory train-scene goal matching; no RL or val-unseen result.",
              "seed": args.seed, "best_step": best_step, "fit_episodes": len(fit["ids"]),
              "fit_pairs": len(fit["pairs"]), "development_scenes": scenes[:5],
              "audit_scenes": scenes[5:], "development": best_development,
              "locked_audit": audited,
              "candidate_gate": {"counterfactual_at_least_0_75": audited["counterfactual_accuracy"] >= .75,
                                 "above_raw_by_0_05": audited["counterfactual_accuracy"] -
                                     audited["raw_backbone_counterfactual_accuracy"] >= .05,
                                 "endpoint_above_start_at_least_0_65":
                                     audited["endpoint_above_start_rate"] >= .65}}
    report["candidate_gate"]["status"] = ("passed" if all(report["candidate_gate"].values())
                                           else "failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": best_state, "feature_dim": fit["images"].shape[-1],
                "seed": args.seed, "best_step": best_step}, args.output_dir / "goal_matcher.pt")
    print(json.dumps({"development": best_development, "locked_audit": audited,
                      "candidate_gate": report["candidate_gate"]}, indent=2))


if __name__ == "__main__":
    main()
