"""CPU probe of a state-dependent, order-antisymmetric motion classifier.

Uses the frozen true-n=4 policy-history cache and privileged R2R-train
distance labels. It is an exploratory representation test, not a reward
or navigation evaluation. Reversing before/after negates its score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import torch
from torch import nn
from torch.nn import functional as F

from fit_group_relative_head import load_part, nested_indices, sample
from history_grounding_lora import digest


class MotionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2048)
        self.linear = nn.Linear(2048, 1, bias=False)
        self.delta = nn.Linear(2048, 64, bias=False)
        self.context = nn.Linear(2048, 64, bias=False)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        before = self.norm(before.float())
        after = self.norm(after.float())
        change = after - before
        context = (after + before) * .5
        return (self.linear(change).squeeze(-1) +
                (self.delta(change) * torch.tanh(self.context(context)))
                .sum(dim=-1) / math.sqrt(64))


def episode_key(row: tuple) -> tuple[str, str]:
    scene, gid = row[:2]
    return scene, gid.split("_e", 1)[1]


def measure(model: MotionHead, rows: list[tuple], label: float) -> dict:
    by_episode = defaultdict(list)
    by_scene = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            before = torch.stack([row[2] for row in batch])
            after = torch.stack([row[3] for row in batch])
            scores = model(before, after).tolist()
            for row, score in zip(batch, scores):
                point = float(score * label > 0) if score else .5
                by_episode[episode_key(row)].append(point)
    for (scene, _), points in by_episode.items():
        by_scene[scene].append(statistics.mean(points))
    return {
        "pairs": len(rows), "episode_groups": len(by_episode),
        "scenes": len(by_scene),
        "pair_accuracy": statistics.mean(
            point for points in by_episode.values() for point in points),
        "episode_macro": statistics.mean(
            statistics.mean(points) for points in by_episode.values()),
        "scene_macro": statistics.mean(
            statistics.mean(points) for points in by_scene.values()),
    }


def evaluate(model: MotionHead, data: dict) -> dict:
    model.eval()
    forward = measure(model, data["forward"], 1.0)
    regression = measure(model, data["regression"], -1.0)
    return {"forward": forward, "regression": regression,
            "balanced_episode_macro": .5 * (
                forward["episode_macro"] + regression["episode_macro"])}


def ordered_temporal(data: dict) -> dict:
    # Existing audited rows store the closer representation before the
    # farther representation. Restore chronological order for a motion
    # classifier rather than training a scalar state potential.
    return {
        "forward": [(scene, gid, farther, closer)
                    for scene, gid, closer, farther in data["forward"]],
        "regression": [(scene, gid, closer, farther)
                       for scene, gid, closer, farther in data["regression"]],
    }


def run_seed(seed: int, fit: dict, dev: dict) -> tuple[dict, dict]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = MotionHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,
                                  weight_decay=.01)
    nested = {kind: nested_indices(fit[kind])
              for kind in ("forward", "regression")}
    history = []
    for epoch in range(1, 13):
        model.train()
        for _ in range(24):
            forward = [fit["forward"][sample(nested["forward"], rng)]
                       for _ in range(64)]
            regression = [fit["regression"][sample(nested["regression"], rng)]
                          for _ in range(64)]
            batch = [(row, 1.0) for row in forward] + \
                [(row, -1.0) for row in regression]
            rng.shuffle(batch)
            before = torch.stack([row[2] for row, _ in batch])
            after = torch.stack([row[3] for row, _ in batch])
            labels = torch.tensor([target for _, target in batch])
            score = model(before, after)
            loss = F.softplus(-labels * score).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite directional loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch in (4, 8, 12):
            metrics = evaluate(model, dev)
            history.append({"epoch": epoch, "development": metrics})
            print(json.dumps({"seed": seed, "epoch": epoch,
                              "development": metrics}), flush=True)
    model.eval()
    # Check the architectural invariant on real development features.
    a = dev["forward"][0][2].unsqueeze(0)
    b = dev["forward"][0][3].unsqueeze(0)
    with torch.inference_mode():
        if not torch.allclose(model(a, b), -model(b, a),
                              atol=1e-4, rtol=1e-5):
            raise ValueError("motion head lost order antisymmetry")
    report = {"seed": seed, "fit": evaluate(model, fit),
              "development": evaluate(model, dev), "history": history}
    weights = {key: value.detach().cpu()
               for key, value in model.state_dict().items()}
    return report, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "turn-root", "cache-root", "cache-audit",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.cache_audit.read_text())
    manifest_sha = digest(args.manifest)
    if manifest.get("schema") != "policy_group_relative_manifest_v1" or \
            audit.get("schema") != "group_relative_state_cache_audit_v1" or \
            audit.get("manifest_sha256") != manifest_sha:
        raise ValueError("frozen group-four source mismatch")
    fit_raw = load_part("fit", manifest, manifest_sha,
                        args.turn_root, args.cache_root, audit["source_id"])
    dev_raw = load_part("development", manifest, manifest_sha,
                        args.turn_root, args.cache_root, audit["source_id"])
    if fit_raw["scenes"] & dev_raw["scenes"]:
        raise ValueError("fit/development scene leakage")
    fit, dev = ordered_temporal(fit_raw), ordered_temporal(dev_raw)
    if any(not fit[kind] or not dev[kind]
           for kind in ("forward", "regression")):
        raise ValueError("directional labels missing")
    runs, weights = [], {}
    for seed in (11, 22, 33):
        report, state = run_seed(seed, fit, dev)
        runs.append(report)
        weights[str(seed)] = state
    by_seed = {str(row["seed"]): row for row in runs}
    gate = {}
    for seed in (11, 22, 33):
        metric = by_seed[str(seed)]["development"]
        gate[str(seed)] = {
            "forward_episode_macro_at_least_70pct": (
                metric["forward"]["episode_macro"] >= .70),
            "regression_episode_macro_at_least_65pct": (
                metric["regression"]["episode_macro"] >= .65),
            "balanced_episode_macro_at_least_70pct": (
                metric["balanced_episode_macro"] >= .70),
        }
    report = {
        "schema": "antisymmetric_motion_cached_group4_dev_v1",
        "manifest_sha256": manifest_sha,
        "cache_audit_sha256": digest(args.cache_audit),
        "source_id": audit["source_id"],
        "fit_groups": fit_raw["groups"],
        "development_groups": dev_raw["groups"],
        "training": "fixed 12 epochs, class/scene/group-balanced sampled logistic loss; no development checkpoint selection",
        "seeds": runs, "directional_checks": gate,
        "all_seed_directional_checks_pass": all(
            all(values.values()) for values in gate.values()),
        "interpretation": (
            "Exploratory reused R2R-train development test. No stationary "
            "or wrong-instruction audit, process reward, online RL, or "
            "val-unseen navigation result."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    torch.save({"schema": "antisymmetric_motion_cached_group4_heads_v1",
                "manifest_sha256": manifest_sha,
                "cache_audit_sha256": digest(args.cache_audit),
                "state_dicts": weights}, args.output / "heads.pt")
    print(json.dumps({"directional_checks": gate,
                      "all_seed_directional_checks_pass":
                      report["all_seed_directional_checks_pass"]}, indent=2))


if __name__ == "__main__":
    main()
