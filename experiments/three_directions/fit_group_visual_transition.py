"""Instruction-attended patch transition reward for group-four VLN.

The frozen SigLIP towers encode real before/after RGB and instruction.
A small trainable module attends to visual patches and predicts a signed
change, with exact antisymmetry under frame reversal. It is selected on
R2R-train development scenes; the locked audit and unseen set remain closed.
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


INTERVALS = ((3, 6), (6, 9), (9, 12))


class VisualTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(768)
        self.text_norm = nn.LayerNorm(768)
        self.query = nn.Linear(768, 64, bias=False)
        self.key = nn.Linear(768, 64, bias=False)
        self.value = nn.Linear(768, 64, bias=False)
        self.delta = nn.Linear(128, 64, bias=False)
        self.context = nn.Linear(128, 64)
        self.readout = nn.Linear(64, 1, bias=False)

    def represent(self, patches: torch.Tensor,
                  text: torch.Tensor) -> torch.Tensor:
        patches = self.visual_norm(patches.float())
        query = self.query(self.text_norm(text.float()))
        keys = self.key(patches)
        values = self.value(patches)
        attention = torch.softmax((keys * query[:, None, :]).sum(-1) / 8,
                                  dim=-1)
        attended = (values * attention[..., None]).sum(1)
        background = values.mean(1)
        return torch.cat((attended, background), dim=-1)

    def forward(self, before: torch.Tensor, after: torch.Tensor,
                text: torch.Tensor) -> torch.Tensor:
        prior = self.represent(before, text)
        next_state = self.represent(after, text)
        difference = next_state - prior
        midpoint = .5 * (next_state + prior)
        odd = torch.tanh(self.delta(difference))
        gate = torch.sigmoid(self.context(midpoint))
        return self.readout(odd * gate).squeeze(-1)


def load_part(part: str, manifest: dict, manifest_sha: str,
              turn_root: Path, cache_root: Path, source_id: str) -> dict:
    local = {"forward": [], "regression": []}
    relative = []
    scenes = set()
    episodes = set()
    for group in manifest["groups"][part]:
        gid = group["group_id"]
        scene = group["scene_id"]
        variants = []
        for variant in range(4):
            record_id = rid({"seed": group["seed"],
                             "episode_id": group["episode_id"],
                             "variant": variant})
            record = json.loads((turn_root / part / "records" /
                                 f"{record_id}.json").read_text())
            cache = torch.load(cache_root / part / "records" /
                               f"{record_id}.pt", map_location="cpu",
                               weights_only=True)
            if (record["record_id"] != record_id or
                    record["manifest_sha256"] != manifest_sha or
                    record["scene_id"] != scene or
                    cache["schema"] != "group_visual_tokens_v1" or
                    cache["manifest_sha256"] != manifest_sha or
                    cache["source_id"] != source_id or
                    cache["record_id"] != record_id or
                    cache["patches"].shape !=
                        (len(cache["anchor_turns"]), 196, 768) or
                    cache["text"].shape != (768,)):
                raise ValueError(f"visual cache mismatch {part}/{record_id}")
            distance = {turn["original_turn_index"]:
                        turn["distance_to_goal_for_label_only"]
                        for turn in record["turns"]}
            states = {turn: (cache["patches"][index], distance[turn])
                      for index, turn in enumerate(cache["anchor_turns"])}
            variants.append((states, cache["text"]))
        scenes.add(scene)
        episodes.add(str(group["episode_id"]))
        for start, end in INTERVALS:
            active = []
            for states, text in variants:
                if start not in states or end not in states:
                    continue
                before, prior_distance = states[start]
                after, next_distance = states[end]
                progress = prior_distance - next_distance
                transition = (before, after, text)
                active.append((transition, progress))
                if progress >= 1.0:
                    local["forward"].append((scene, gid, transition))
                elif progress <= -1.0:
                    local["regression"].append((scene, gid, transition))
            for index, (left, left_progress) in enumerate(active):
                for right, right_progress in active[index + 1:]:
                    if abs(left_progress - right_progress) < 1.0:
                        continue
                    better, worse = (left, right) if left_progress > \
                                    right_progress else (right, left)
                    relative.append((scene, gid, better, worse))
    if not relative or not local["forward"] or not local["regression"]:
        raise ValueError(f"insufficient visual transition labels {part}")
    return {"relative": relative, **local, "scenes": scenes,
            "episodes": episodes, "groups": len(manifest["groups"][part])}


def nested(rows: list[tuple]) -> dict:
    result = defaultdict(lambda: defaultdict(list))
    for index, (scene, gid, *_) in enumerate(rows):
        result[scene][gid].append(index)
    return result


def choose(buckets: dict, rng: random.Random) -> int:
    scene = rng.choice(list(buckets))
    gid = rng.choice(list(buckets[scene]))
    return rng.choice(buckets[scene][gid])


def score_transitions(model: VisualTransition, rows: list[tuple],
                      device: torch.device) -> torch.Tensor:
    before = torch.stack([item[0] for item in rows]).to(device)
    after = torch.stack([item[1] for item in rows]).to(device)
    text = torch.stack([item[2] for item in rows]).to(device)
    return model(before, after, text)


def evaluate(model: VisualTransition, data: dict,
             device: torch.device) -> dict:
    model.eval()
    report = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for kind in ("forward", "regression"):
            rows = data[kind]
            correct = 0
            for start in range(0, len(rows), 32):
                scores = score_transitions(model,
                    [row[2] for row in rows[start:start + 32]], device)
                correct += int(((scores > 0) if kind == "forward" else
                                (scores < 0)).sum())
            report[kind] = {"pairs": len(rows),
                            "groups": len({row[1] for row in rows}),
                            "accuracy": correct / len(rows)}
        rows = data["relative"]
        correct = 0
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            better = score_transitions(model, [row[2] for row in batch], device)
            worse = score_transitions(model, [row[3] for row in batch], device)
            correct += int((better > worse).sum())
        report["relative"] = {"pairs": len(rows),
                              "groups": len({row[1] for row in rows}),
                              "accuracy": correct / len(rows)}
    report["balanced_local_accuracy"] = .5 * (
        report["forward"]["accuracy"] + report["regression"]["accuracy"])
    return report


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
            audit["schema"] != "group_visual_token_cache_audit_v1" or
            audit["manifest_sha256"] != manifest_sha or
            audit["source_id"] != args.source_id):
        raise ValueError("manifest/visual cache audit mismatch")
    fit = load_part("fit", manifest, manifest_sha, args.turn_root,
                    args.cache_root, args.source_id)
    dev = load_part("development", manifest, manifest_sha, args.turn_root,
                    args.cache_root, args.source_id)
    if fit["scenes"] & dev["scenes"] or fit["episodes"] & dev["episodes"]:
        raise ValueError("fit/development leakage")
    rng = random.Random(11)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.set_num_threads(6)
    device = torch.device("cuda")
    model = VisualTransition().to(device)
    with torch.inference_mode():
        a, b = torch.randn(2, 196, 768, device=device), \
               torch.randn(2, 196, 768, device=device)
        t = torch.randn(2, 768, device=device)
        if (model(a, b, t) + model(b, a, t)).abs().max() > 1e-5:
            raise ValueError("visual score is not antisymmetric")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-3)
    buckets = {name: nested(fit[name]) for name in
               ("forward", "regression", "relative")}
    best = None
    history = []
    started = time.time()
    for epoch in range(1, 21):
        model.train()
        for _ in range(16):
            positive = [fit["forward"][choose(buckets["forward"], rng)]
                        for _ in range(16)]
            negative = [fit["regression"][choose(buckets["regression"], rng)]
                        for _ in range(16)]
            comparison = [fit["relative"][choose(buckets["relative"], rng)]
                          for _ in range(32)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward_score = score_transitions(model,
                    [row[2] for row in positive], device)
                regression_score = score_transitions(model,
                    [row[2] for row in negative], device)
                better_score = score_transitions(model,
                    [row[2] for row in comparison], device)
                worse_score = score_transitions(model,
                    [row[3] for row in comparison], device)
                loss = .5 * (F.softplus(-forward_score).mean() +
                             F.softplus(regression_score).mean()) + \
                       F.softplus(-(better_score - worse_score)).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite visual loss epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metric = evaluate(model, dev, device)
        history.append({"epoch": epoch, **metric})
        print(json.dumps(history[-1]), flush=True)
        quality = (min(metric["relative"]["accuracy"] - .70,
                       metric["forward"]["accuracy"] - .70,
                       metric["regression"]["accuracy"] - .60),
                   metric["balanced_local_accuracy"], -epoch)
        if best is None or quality > best[0]:
            best = (quality, epoch,
                    copy.deepcopy({key: value.detach().cpu() for key, value
                                   in model.state_dict().items()}), metric)
        if epoch - best[1] >= 5:
            break
    model.load_state_dict(best[2])
    selected_fit = evaluate(model, fit, device)
    selected_dev = best[3]
    gate = {"relative_pairs_at_least_100":
                selected_dev["relative"]["pairs"] >= 100,
            "relative_groups_at_least_20":
                selected_dev["relative"]["groups"] >= 20,
            "forward_pairs_at_least_100":
                selected_dev["forward"]["pairs"] >= 100,
            "regression_pairs_at_least_50":
                selected_dev["regression"]["pairs"] >= 50,
            "relative_accuracy_at_least_0_70":
                selected_dev["relative"]["accuracy"] >= .70,
            "forward_accuracy_at_least_0_70":
                selected_dev["forward"]["accuracy"] >= .70,
            "regression_accuracy_at_least_0_60":
                selected_dev["regression"]["accuracy"] >= .60}
    report = {"schema": "group_visual_transition_development_v1",
              "interpretation": "Exploratory patch-level fit on opened development scenes; locked audit unopened.",
              "seed": 11, "selected_epoch": best[1],
              "manifest_sha256": manifest_sha,
              "source_id": args.source_id,
              "cache_audit_sha256": digest(args.cache_audit),
              "fit_scenes": len(fit["scenes"]),
              "development_scenes": len(dev["scenes"]),
              "fit": selected_fit, "development": selected_dev,
              "development_gate": gate, "passed": all(gate.values()),
              "history": history, "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "group_visual_transition_seed11_v1",
                "manifest_sha256": manifest_sha,
                "source_id": args.source_id,
                "selected_epoch": best[1], "state_dict": best[2]},
               args.output / "head.pt")
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_epoch": best[1],
                      "development": selected_dev,
                      "development_gate": gate,
                      "passed": report["passed"]}, indent=2), flush=True)
    if not report["passed"]:
        (args.output / "development_rejected").write_text("\n")


if __name__ == "__main__":
    main()
