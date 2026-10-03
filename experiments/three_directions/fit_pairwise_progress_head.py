"""Fit an antisymmetric local-progress head on cached navigation-SFT states.

Balanced real forward/regression pairs come from R2R-train scenes only.
No audit records or val-unseen outcomes are read for fitting or selection.
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

from history_grounding_lora import digest, rid, signed_pairs


class AntisymmetricChange(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2048)
        self.net = nn.Sequential(nn.Linear(2048, 128, bias=False),
                                 nn.Tanh(), nn.Linear(128, 1, bias=False))

    def forward(self, before: torch.Tensor,
                after: torch.Tensor) -> torch.Tensor:
        difference = self.norm(after.float()) - self.norm(before.float())
        return self.net(difference).squeeze(-1)


def load_pairs(part: str, policy_manifest: dict, manifest_sha: str,
               policy_root: Path, cache_root: Path, source_id: str) -> dict:
    rows = []
    scenes = set()
    episodes = set()
    for plan in policy_manifest["selected"][part]:
        record_id = rid(plan)
        record = json.loads((policy_root / part / "records" /
                             f"{record_id}.json").read_text())
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        signed = signed_pairs(distances)
        selected = {"forward": signed["forward"][:2],
                    "backward": signed["backward"]}
        if not any(selected.values()):
            continue
        cache = torch.load(cache_root / part / "records" / f"{record_id}.pt",
                           map_location="cpu", weights_only=True)
        if cache["schema"] != "pairwise_policy_state_cache_v1" or \
                cache["source_id"] != source_id or \
                cache["record_id"] != record_id or \
                cache["policy_manifest_sha256"] != manifest_sha or \
                not torch.isfinite(cache["hidden"]).all():
            raise ValueError(f"invalid cached trajectory {part}/{record_id}")
        vectors = dict(zip(cache["indices"], cache["hidden"].float()))
        for category, afters in selected.items():
            for after in afters:
                if after - 1 not in vectors or after not in vectors:
                    raise ValueError(f"missing pair state {part}/{record_id}/{after}")
                rows.append({"scene": record["scene_id"],
                             "episode": str(record["episode_id"]),
                             "record_id": record_id, "category": category,
                             "before": vectors[after - 1],
                             "after": vectors[after]})
                scenes.add(record["scene_id"])
                episodes.add(str(record["episode_id"]))
    if not rows:
        raise ValueError(f"empty {part} pair set")
    return {"rows": rows, "scenes": scenes, "episodes": episodes}


def indices(rows: list[dict]) -> dict:
    nested = {category: defaultdict(lambda: defaultdict(list))
              for category in ("forward", "backward")}
    for index, row in enumerate(rows):
        nested[row["category"]][row["scene"]][row["episode"]].append(index)
    if any(not nested[category] for category in nested):
        raise ValueError("missing positive or negative local progress")
    return nested


def sample(nested: dict, category: str, rng: random.Random) -> int:
    scene = rng.choice(list(nested[category]))
    episode = rng.choice(list(nested[category][scene]))
    return rng.choice(nested[category][scene][episode])


def measure(model: AntisymmetricChange, rows: list[dict]) -> dict:
    model.eval()
    correct = {"forward": 0, "backward": 0}
    count = {"forward": 0, "backward": 0}
    margins = []
    with torch.inference_mode():
        for row in rows:
            value = float(model(row["before"], row["after"]))
            predicted = value > 0 if row["category"] == "forward" else value < 0
            correct[row["category"]] += predicted
            count[row["category"]] += 1
            margins.append(abs(value))
    result = {f"{kind}_accuracy": correct[kind] / count[kind]
              for kind in count}
    result.update({f"{kind}_pairs": count[kind] for kind in count})
    result["balanced_accuracy"] = .5 * (
        result["forward_accuracy"] + result["backward_accuracy"])
    result["median_abs_margin"] = float(torch.tensor(margins).median())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.policy_manifest.read_text())
    manifest_sha = digest(args.policy_manifest)
    audit = json.loads(args.cache_audit.read_text())
    if audit["schema"] != "pairwise_policy_state_cache_audit_v1" or \
            audit["policy_manifest_sha256"] != manifest_sha or \
            audit["source_id"] != args.source_id:
        raise ValueError("feature cache not audited")
    fit = load_pairs("fit", manifest, manifest_sha,
                     args.policy_root, args.cache_root, args.source_id)
    dev = load_pairs("development", manifest, manifest_sha,
                     args.policy_root, args.cache_root, args.source_id)
    if fit["scenes"] & dev["scenes"] or fit["episodes"] & dev["episodes"]:
        raise ValueError("fit/development leakage")
    rng = random.Random(11)
    torch.manual_seed(11)
    torch.set_num_threads(6)
    model = AntisymmetricChange()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-3)
    nested = indices(fit["rows"])
    best = None
    history = []
    started = time.time()
    for epoch in range(1, 51):
        model.train()
        for _ in range(32):
            chosen = [sample(nested, "forward", rng) for _ in range(64)] + [
                sample(nested, "backward", rng) for _ in range(64)]
            rng.shuffle(chosen)
            before = torch.stack([fit["rows"][i]["before"] for i in chosen])
            after = torch.stack([fit["rows"][i]["after"] for i in chosen])
            signs = torch.tensor([1.0 if fit["rows"][i]["category"] ==
                                  "forward" else -1.0 for i in chosen])
            values = model(before, after)
            loss = F.softplus(-signs * values).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = measure(model, dev["rows"])
        row = {"epoch": epoch, **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        quality = (min(metrics["balanced_accuracy"] - .75,
                       metrics["backward_accuracy"] - .60),
                   metrics["balanced_accuracy"], -epoch)
        if best is None or quality > best[0]:
            best = (quality, epoch,
                    copy.deepcopy({key: value.detach().cpu() for key, value
                                   in model.state_dict().items()}), metrics)
        if epoch - best[1] >= 8:
            break
    if best is None:
        raise ValueError("no selected head")
    report = {"schema": "pairwise_progress_head_development_v1",
              "seed": 11, "source_id": args.source_id,
              "policy_manifest_sha256": manifest_sha,
              "cache_audit_sha256": digest(args.cache_audit),
              "selected_epoch": best[1],
              "fit_pairs": {"forward": sum(row["category"] == "forward" for row in fit["rows"]),
                            "backward": sum(row["category"] == "backward" for row in fit["rows"])},
              "fit_unique_episodes": len(fit["episodes"]),
              "development_unique_episodes": len(dev["episodes"]),
              "fit_scenes": len(fit["scenes"]),
              "development_scenes": len(dev["scenes"]),
              "development": best[3],
              "development_gate": {
                  "balanced_accuracy_at_least_0_75":
                      best[3]["balanced_accuracy"] >= .75,
                  "regression_accuracy_at_least_0_60":
                      best[3]["backward_accuracy"] >= .60},
              "history": history, "elapsed_seconds": time.time() - started,
              "interpretation": "Development-only local pairwise screen; locked audit unopened."}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "pairwise_progress_head_seed11_v1",
                "source_id": args.source_id,
                "policy_manifest_sha256": manifest_sha,
                "selected_epoch": best[1], "state_dict": best[2]},
               args.output / "head.pt")
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_epoch": best[1],
                      "development": best[3],
                      "development_gate": report["development_gate"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
