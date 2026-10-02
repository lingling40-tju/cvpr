"""Train a trajectory-conditioned instruction matcher on frozen RGB features.

This deliberately tests a different representation from single-frame ordinal
progress: a causal visual history is contrasted with natural instructions
sharing the same start pose. It produces offline diagnostics only, no RL reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMatcher(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(dim * 3, 128), nn.GELU())
        self.text = nn.Sequential(nn.Linear(dim, 128), nn.GELU())
        self.gru = nn.GRU(256, 128, batch_first=True)
        self.score = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 128),
                                   nn.GELU(), nn.Linear(128, 1))

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        start = images[:, :1, :]
        prior = torch.cat((images[:, :1, :], images[:, :-1, :]), dim=1)
        visual = self.frame(torch.cat((images, images - start, images - prior), dim=-1))
        language = self.text(text)
        state, _ = self.gru(torch.cat((visual,
                                      language[:, None, :].expand(-1, images.shape[1], -1)), dim=-1))
        repeated_language = language[:, None, :].expand_as(state)
        return self.score(torch.cat((state, repeated_language,
                                     state * repeated_language), dim=-1)).squeeze(-1).sigmoid()


def load_sequences(path: Path, manifest: dict, subset: str, device: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected = manifest["subsets"][subset]["episode_ids"]
    ids = data["episode_ids"].tolist()
    if data["subset"] != subset or ids != expected or len(data["scene_ids"]) != len(ids):
        raise ValueError(f"feature coverage/split mismatch for {subset}")
    by_episode = defaultdict(list)
    for index, episode_id in enumerate(data["frame_episode_ids"].tolist()):
        by_episode[episode_id].append(index)
    sequences = []
    fractions = []
    for episode_id in ids:
        indices = sorted(by_episode[episode_id], key=lambda j: data["frame_offsets"][j].item())
        if len(indices) != 6:
            raise ValueError(f"expected six frames for episode {episode_id}")
        sequences.append(data["images"][indices])
        fractions.append(data["progress_fractions"][indices])
    if not torch.isfinite(data["images"]).all() or not torch.isfinite(data["texts"]).all():
        raise ValueError("nonfinite frozen features")
    id_to_index = {episode_id: i for i, episode_id in enumerate(ids)}
    pairs = []
    for pair in manifest["subsets"][subset]["pairs"]:
        a, b = pair["left"], pair["right"]
        if a in id_to_index and b in id_to_index:
            pairs.append((id_to_index[a], id_to_index[b]))
    return {"ids": ids, "scenes": data["scene_ids"],
            "images": torch.stack(sequences).to(device),
            "texts": data["texts"].to(device),
            "fractions": torch.stack(fractions).to(device),
            "pairs": pairs, "backbone": data["backbone"],
            "model_config_sha256": data["model_config_sha256"]}


def select_scenes(data: dict, scenes: set[str]) -> dict:
    selected = [i for i, scene in enumerate(data["scenes"]) if scene in scenes]
    if not selected or set(data["scenes"][i] for i in selected) != scenes:
        raise ValueError("scene selection is incomplete")
    remap = {old: new for new, old in enumerate(selected)}
    pairs = [(remap[a], remap[b]) for a, b in data["pairs"] if a in remap and b in remap]
    if not pairs:
        raise ValueError("scene split has no same-start instruction pairs")
    return {"ids": [data["ids"][i] for i in selected],
            "scenes": [data["scenes"][i] for i in selected],
            "images": data["images"][selected], "texts": data["texts"][selected],
            "fractions": data["fractions"][selected], "pairs": pairs,
            "backbone": data["backbone"],
            "model_config_sha256": data["model_config_sha256"]}


@torch.no_grad()
def evaluate(model: TemporalMatcher, data: dict) -> dict:
    model.eval()
    images, texts = data["images"], data["texts"]
    own = torch.cat([model(images[i:i + 64], texts[i:i + 64])
                     for i in range(0, len(images), 64)], dim=0)
    ord_hits = torch.stack([(own[:, j] > own[:, i]).float()
                            for i in range(6) for j in range(i + 1, 6)], dim=1)
    pairs = torch.tensor(data["pairs"], dtype=torch.long, device=images.device)
    a, b = pairs.T
    a_wrong = model(images[a], texts[b])
    b_wrong = model(images[b], texts[a])
    correct = torch.cat((own[a, -2:], own[b, -2:])).flatten()
    wrong = torch.cat((a_wrong[:, -2:], b_wrong[:, -2:])).flatten()
    # Frozen backbone's endpoint comparison uses the identical pair examples.
    raw_correct = torch.cat(((images[a, -2:] * texts[a, None, :]).sum(-1),
                             (images[b, -2:] * texts[b, None, :]).sum(-1))).flatten()
    raw_wrong = torch.cat(((images[a, -2:] * texts[b, None, :]).sum(-1),
                           (images[b, -2:] * texts[a, None, :]).sum(-1))).flatten()
    raw_own = (images * texts[:, None, :]).sum(-1)
    raw_order = torch.stack([(raw_own[:, j] > raw_own[:, i]).float()
                             for i in range(6) for j in range(i + 1, 6)], dim=1)
    stutter = model(torch.cat((images, images[:, -1:, :]), dim=1), texts)[:, -1] - own[:, -1]
    return {"episodes": len(images), "scenes": len(set(data["scenes"])),
            "pairs": len(data["pairs"]), "ordered_comparisons": ord_hits.numel(),
            "counterfactual_comparisons": len(correct),
            "ordinal_accuracy": ord_hits.mean().item(),
            "counterfactual_accuracy": (correct > wrong).float().mean().item(),
            "counterfactual_margin_mean": (correct - wrong).mean().item(),
            "progress_mse": F.mse_loss(own, data["fractions"]).item(),
            "stutter_positive_increase_mean": stutter.relu().mean().item(),
            "raw_backbone_ordinal_accuracy": raw_order.mean().item(),
            "raw_backbone_counterfactual_accuracy": (raw_correct > raw_wrong).float().mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(args.manifest.read_text())
    fit_all = load_sequences(args.fit_features, manifest, "fit", device)
    calibration = load_sequences(args.calibration_features, manifest, "calibration", device)
    if fit_all["backbone"] != calibration["backbone"] or \
            fit_all["model_config_sha256"] != calibration["model_config_sha256"]:
        raise ValueError("fit/calibration backbone mismatch")
    pair_scenes = sorted({pair["scene"] for pair in manifest["subsets"]["fit"]["pairs"]},
                         key=lambda scene: hashlib.sha256(("temporal-v1:" + scene).encode()).hexdigest())
    if len(pair_scenes) < 15:
        raise ValueError("insufficient natural instruction-pair scenes")
    development_scenes = set(pair_scenes[:5])
    audit_scenes = set(pair_scenes[5:10])
    training_scenes = set(manifest["fit_scenes"]) - development_scenes - audit_scenes
    train = select_scenes(fit_all, training_scenes)
    development = select_scenes(fit_all, development_scenes)
    audit = select_scenes(fit_all, audit_scenes)
    if set(train["ids"]) & (set(development["ids"]) | set(audit["ids"])):
        raise ValueError("train/development/audit episode overlap")
    model = TemporalMatcher(train["images"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pair_index = torch.tensor(train["pairs"], dtype=torch.long, device=device)
    best, best_step, best_state, best_development = float("-inf"), 0, None, None
    for step in range(1, args.steps + 1):
        model.train()
        ids = torch.randint(len(train["ids"]), (32,), generator=generator, device=device)
        pair_ids = torch.randint(len(pair_index), (16,), generator=generator, device=device)
        a, b = pair_index[pair_ids].T
        own = model(train["images"][ids], train["texts"][ids])
        own_a = model(train["images"][a], train["texts"][a])
        own_b = model(train["images"][b], train["texts"][b])
        wrong_a = model(train["images"][a], train["texts"][b])
        wrong_b = model(train["images"][b], train["texts"][a])
        margin = torch.cat((own_a[:, -2:] - wrong_a[:, -2:],
                            own_b[:, -2:] - wrong_b[:, -2:]), dim=0)
        stutter = model(torch.cat((train["images"][ids],
                                   train["images"][ids, -1:, :]), dim=1),
                        train["texts"][ids])[:, -1] - own[:, -1]
        loss = (0.3 * F.mse_loss(own, train["fractions"][ids])
                + 0.5 * F.softplus(0.08 - (own[:, 1:] - own[:, :-1])).mean()
                + 2.0 * F.softplus(0.15 - margin).mean()
                + 2.0 * stutter.relu().square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            metrics = evaluate(model, development)
            score = (metrics["ordinal_accuracy"] + 1.5 * metrics["counterfactual_accuracy"]
                     - metrics["progress_mse"] - metrics["stutter_positive_increase_mean"])
            if score > best:
                best, best_step, best_development = score, step, metrics
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            print(json.dumps({"step": step, "train_loss": loss.item(), "development": metrics}), flush=True)
            if step - best_step >= 150:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    audit_metrics = evaluate(model, audit)
    secondary_metrics = evaluate(model, calibration)
    report = {
        "interpretation": "Offline train-scene trajectory matcher; no RL or val-unseen result.",
        "seed": args.seed, "best_step": best_step, "backbone": train["backbone"],
        "train_episodes": len(train["ids"]), "development_episodes": len(development["ids"]),
        "locked_audit_episodes": len(audit["ids"]), "secondary_calibration_episodes": len(calibration["ids"]),
        "development_scenes": sorted(development_scenes),
        "locked_audit_scenes": sorted(audit_scenes),
        "development": best_development, "locked_audit": audit_metrics,
        "secondary_calibration": secondary_metrics,
        "candidate_gate": {
            "ordinal_at_least_0_70": audit_metrics["ordinal_accuracy"] >= 0.70,
            "counterfactual_at_least_0_75": audit_metrics["counterfactual_accuracy"] >= 0.75,
            "counterfactual_above_raw_by_0_05":
                audit_metrics["counterfactual_accuracy"] -
                audit_metrics["raw_backbone_counterfactual_accuracy"] >= 0.05,
            "stutter_increase_at_most_0_02":
                audit_metrics["stutter_positive_increase_mean"] <= 0.02,
        }}
    report["candidate_gate"]["status"] = (
        "passed" if all(report["candidate_gate"][key]
                        for key in ("ordinal_at_least_0_70", "counterfactual_at_least_0_75",
                                    "counterfactual_above_raw_by_0_05", "stutter_increase_at_most_0_02"))
        else "failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feature_dim": train["images"].shape[-1],
                "backbone": train["backbone"], "seed": args.seed,
                "best_step": best_step}, args.output_dir / "temporal_head.pt")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"locked_audit": audit_metrics,
                      "candidate_gate": report["candidate_gate"]}, indent=2))


if __name__ == "__main__":
    main()
