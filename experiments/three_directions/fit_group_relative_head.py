"""Fit a four-rollout relative progress head on frozen policy histories.

For each R2R-train episode, four policy rollouts share the instruction and
start. Same-turn comparisons cancel these nuisances; temporal comparisons
require the learned score to respond to actual forward and backward motion.
Only fit/development scenes are opened here. Audit scenes stay locked.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from history_grounding_lora import digest, rid


class GroupRelativeScore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2048)
        self.net = nn.Sequential(nn.Linear(2048, 128, bias=False),
                                 nn.Tanh(), nn.Linear(128, 1, bias=False))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(hidden.float())).squeeze(-1)


def load_part(part: str, manifest: dict, manifest_sha: str, turn_root: Path,
              cache_root: Path, source_id: str) -> dict:
    groups = []
    scenes = set()
    episodes = set()
    for group in manifest["groups"][part]:
        gid = group["group_id"]
        variants = []
        for variant in range(4):
            plan = {"seed": group["seed"], "episode_id": group["episode_id"],
                    "variant": variant}
            record_id = rid(plan)
            record = json.loads((turn_root / part / "records" /
                                 f"{record_id}.json").read_text())
            cache = torch.load(cache_root / part / "records" /
                               f"{record_id}.pt", map_location="cpu",
                               weights_only=True)
            if (record["record_id"] != record_id or
                    record["manifest_sha256"] != manifest_sha or
                    cache["schema"] != "group_relative_state_cache_v1" or
                    cache["manifest_sha256"] != manifest_sha or
                    cache["source_id"] != source_id or
                    cache["record_id"] != record_id or
                    record["scene_id"] != group["scene_id"] or
                    cache["hidden"].shape[0] != len(cache["anchor_turns"])):
                raise ValueError(f"invalid group cache {part}/{record_id}")
            distances = {turn["original_turn_index"]:
                         turn["distance_to_goal_for_label_only"]
                         for turn in record["turns"]}
            states = {turn: (cache["hidden"][index].float(), distances[turn])
                      for index, turn in enumerate(cache["anchor_turns"])}
            variants.append(states)
        scenes.add(group["scene_id"])
        episodes.add(str(group["episode_id"]))
        groups.append((group, variants))
    if len(groups) != manifest["targets"][part]:
        raise ValueError(f"incomplete group set {part}")
    same_turn, forward, regression = [], [], []
    for group, variants in groups:
        scene = group["scene_id"]
        gid = group["group_id"]
        for turn in (3, 6, 9, 12):
            active = [(variant, states[turn]) for variant, states in
                      enumerate(variants) if turn in states]
            for index, (left_variant, (left_h, left_d)) in enumerate(active):
                for right_variant, (right_h, right_d) in active[index + 1:]:
                    if abs(left_d - right_d) < 1.0:
                        continue
                    better, worse = (left_h, right_h) if left_d < right_d else \
                                    (right_h, left_h)
                    same_turn.append((scene, gid, better, worse))
        for variant, states in enumerate(variants):
            for before_turn, after_turn in zip(sorted(states), sorted(states)[1:]):
                before_h, before_d = states[before_turn]
                after_h, after_d = states[after_turn]
                if before_d - after_d >= 1.0:
                    forward.append((scene, gid, after_h, before_h))
                elif after_d - before_d >= 1.0:
                    regression.append((scene, gid, before_h, after_h))
    return {"same_turn": same_turn, "forward": forward,
            "regression": regression, "scenes": scenes, "episodes": episodes,
            "groups": len(groups)}


def nested_indices(rows: list[tuple]) -> dict:
    nested = defaultdict(lambda: defaultdict(list))
    for index, (scene, gid, _, _) in enumerate(rows):
        nested[scene][gid].append(index)
    if not nested:
        raise ValueError("empty comparison set")
    return nested


def sample(nested: dict, rng: random.Random) -> int:
    scene = rng.choice(list(nested))
    gid = rng.choice(list(nested[scene]))
    return rng.choice(nested[scene][gid])


def measure(model: GroupRelativeScore, rows: list[tuple]) -> dict:
    model.eval()
    correct = 0
    margins = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            better = torch.stack([row[2] for row in batch])
            worse = torch.stack([row[3] for row in batch])
            delta = model(better) - model(worse)
            correct += int((delta > 0).sum())
            margins.extend(delta.tolist())
    return {"pairs": len(rows), "accuracy": correct / len(rows),
            "median_signed_margin": float(torch.tensor(margins).median()),
            "groups": len({row[1] for row in rows})}


def evaluate(model: GroupRelativeScore, data: dict) -> dict:
    scores = {kind: measure(model, data[kind]) for kind in
              ("same_turn", "forward", "regression")}
    scores["balanced_temporal_accuracy"] = .5 * (
        scores["forward"]["accuracy"] + scores["regression"]["accuracy"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    audit = json.loads(args.cache_audit.read_text())
    if (manifest["schema"] != "policy_group_relative_manifest_v1" or
            audit["schema"] != "group_relative_state_cache_audit_v1" or
            audit["manifest_sha256"] != manifest_sha or
            audit["source_id"] != args.source_id):
        raise ValueError("cache audit or manifest mismatch")
    fit = load_part("fit", manifest, manifest_sha, args.turn_root,
                    args.cache_root, args.source_id)
    dev = load_part("development", manifest, manifest_sha, args.turn_root,
                    args.cache_root, args.source_id)
    if fit["scenes"] & dev["scenes"] or fit["episodes"] & dev["episodes"]:
        raise ValueError("fit/development leakage")
    for data in (fit, dev):
        if any(not data[key] for key in ("same_turn", "forward", "regression")):
            raise ValueError("insufficient progress comparison coverage")
    rng = random.Random(11)
    torch.manual_seed(11)
    torch.set_num_threads(6)
    model = GroupRelativeScore()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-3)
    nested = {kind: nested_indices(fit[kind]) for kind in
              ("same_turn", "forward", "regression")}
    best = None
    history = []
    started = time.time()
    for epoch in range(1, 31):
        model.train()
        for _ in range(32):
            selected = [fit["same_turn"][sample(nested["same_turn"], rng)]
                        for _ in range(64)]
            selected += [fit["forward"][sample(nested["forward"], rng)]
                         for _ in range(32)]
            selected += [fit["regression"][sample(nested["regression"], rng)]
                         for _ in range(32)]
            rng.shuffle(selected)
            better = torch.stack([row[2] for row in selected])
            worse = torch.stack([row[3] for row in selected])
            loss = F.softplus(-(model(better) - model(worse))).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite group relative loss epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, dev)
        history.append({"epoch": epoch, **metrics})
        print(json.dumps({"epoch": epoch, **metrics}), flush=True)
        quality = (min(metrics["same_turn"]["accuracy"] - .70,
                       metrics["forward"]["accuracy"] - .70,
                       metrics["regression"]["accuracy"] - .60),
                   metrics["balanced_temporal_accuracy"], -epoch)
        if best is None or quality > best[0]:
            best = (quality, epoch,
                    copy.deepcopy({key: value.detach().cpu() for key, value
                                   in model.state_dict().items()}), metrics)
        if epoch - best[1] >= 6:
            break
    model.load_state_dict(best[2])
    selected_fit = evaluate(model, fit)
    selected_dev = best[3]
    gate = {"same_turn_pairs_at_least_100":
                selected_dev["same_turn"]["pairs"] >= 100,
            "same_turn_groups_at_least_20":
                selected_dev["same_turn"]["groups"] >= 20,
            "forward_pairs_at_least_100":
                selected_dev["forward"]["pairs"] >= 100,
            "regression_pairs_at_least_50":
                selected_dev["regression"]["pairs"] >= 50,
            "same_turn_accuracy_at_least_0_70":
                selected_dev["same_turn"]["accuracy"] >= .70,
            "forward_accuracy_at_least_0_70":
                selected_dev["forward"]["accuracy"] >= .70,
            "regression_accuracy_at_least_0_60":
                selected_dev["regression"]["accuracy"] >= .60}
    report = {"schema": "group_relative_head_development_v1",
              "seed": 11, "manifest_sha256": manifest_sha,
              "source_id": args.source_id,
              "cache_audit_sha256": digest(args.cache_audit),
              "selected_epoch": best[1],
              "fit_scenes": len(fit["scenes"]),
              "development_scenes": len(dev["scenes"]),
              "fit_groups": fit["groups"],
              "development_groups": dev["groups"],
              "fit": selected_fit, "development": selected_dev,
              "development_gate": gate,
              "passed": all(gate.values()), "history": history,
              "elapsed_seconds": time.time() - started,
              "interpretation": "Train-scene development gate only; locked audit and val-unseen unopened."}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "group_relative_head_seed11_v1",
                "source_id": args.source_id,
                "manifest_sha256": manifest_sha,
                "selected_epoch": best[1], "state_dict": best[2]},
               args.output / "head.pt")
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_epoch": best[1], "development": selected_dev,
                      "development_gate": gate, "passed": report["passed"]},
                     indent=2), flush=True)
    if not report["passed"]:
        (args.output / "development_rejected").write_text("\n")


if __name__ == "__main__":
    main()
