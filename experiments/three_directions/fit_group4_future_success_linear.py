"""Screen a frozen-history, group-four future-success value readout.

At preterminal turns 3/6, successful and unsuccessful rollouts from the
same four-rollout episode form pairwise outcome labels. No simulator
distance or terminal status is passed to the readout as an input.
Only fit/development scene partitions are opened here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F


SUCCESS = "successfully reached the goal."
FAILURES = {"stopped but goal not reached.", "number of turns exceeded."}
ANCHORS = (3, 6)
L2 = .01


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_part(manifest: dict, part: str, turn_root: Path, state_root: Path,
              manifest_sha: str, source_id: str) -> tuple[dict, list[dict]]:
    groups = defaultdict(list)
    states = []
    for plan in manifest["selected"][part]:
        rid = f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"
        turn = json.loads((turn_root / part / "records" / f"{rid}.json").read_text())
        cache = torch.load(state_root / part / "records" / f"{rid}.pt",
                           map_location="cpu", weights_only=True)
        if turn["record_id"] != rid or turn["manifest_sha256"] != manifest_sha or \
                turn["scene_id"] != plan["scene_id"] or \
                turn["terminal_mode"] != plan["terminal_mode"] or \
                cache["record_id"] != rid or cache["manifest_sha256"] != manifest_sha or \
                cache["source_id"] != source_id or \
                cache["hidden"].shape != (len(cache["anchor_turns"]), 2048) or \
                not bool(torch.isfinite(cache["hidden"]).all()):
            raise ValueError(f"invalid history source {part}/{rid}")
        last_motion_turn = max(row["original_turn_index"] for row in turn["turns"])
        available = {}
        for anchor, hidden in zip(cache["anchor_turns"], cache["hidden"]):
            if anchor in ANCHORS and anchor < last_motion_turn:
                vector = hidden.float()
                available[anchor] = vector
                states.append(vector)
        groups[(int(plan["seed"]), str(plan["episode_id"]))].append({
            "scene_id": plan["scene_id"], "terminal_mode": plan["terminal_mode"],
            "available": available})
    if len(groups) * 4 != len(manifest["selected"][part]) or \
            any(len(group) != 4 for group in groups.values()):
        raise ValueError(f"incomplete group-four part {part}")
    pairs = []
    for (seed, eid), group in groups.items():
        if len({x["scene_id"] for x in group}) != 1:
            raise ValueError(f"mixed group scenes {part}/{seed}/{eid}")
        for anchor in ANCHORS:
            good = [x["available"][anchor] for x in group
                    if x["terminal_mode"] == SUCCESS and anchor in x["available"]]
            bad = [x["available"][anchor] for x in group
                   if x["terminal_mode"] in FAILURES and anchor in x["available"]]
            for positive in good:
                for negative in bad:
                    pairs.append({"group_id": f"s{seed}_e{eid}", "episode_id": eid,
                                  "scene_id": group[0]["scene_id"],
                                  "anchor": anchor, "difference": positive - negative})
    return {"trajectories": len(manifest["selected"][part]),
            "groups": len(groups), "states": len(states),
            "scenes": len({x["scene_id"] for g in groups.values() for x in g}),
            "matched_pairs": len(pairs),
            "matched_groups": len({x["group_id"] for x in pairs}),
            "by_anchor": {str(a): {"pairs": sum(x["anchor"] == a for x in pairs),
                                   "groups": len({x["group_id"] for x in pairs
                                                  if x["anchor"] == a})}
                          for a in ANCHORS}}, pairs


def metrics(pairs: list[dict], scores: torch.Tensor) -> dict:
    correct = [int(value > 0) for value in scores.tolist()]
    by_group = defaultdict(list)
    by_anchor = defaultdict(list)
    for pair, result in zip(pairs, correct):
        by_group[pair["group_id"]].append(result)
        by_anchor[pair["anchor"]].append(result)
    rng = random.Random(11)
    clusters = list(by_group.values())
    bootstrap = []
    for _ in range(5000):
        selected = [rng.choice(clusters) for _ in clusters]
        bootstrap.append(sum(map(sum, selected)) / sum(map(len, selected)))
    bootstrap.sort()
    return {"pairs": len(pairs), "correct": sum(correct),
            "accuracy": sum(correct) / len(correct),
            "groups": len(by_group),
            "group_macro_accuracy": sum(sum(x) / len(x) for x in by_group.values()) / len(by_group),
            "group_cluster_bootstrap_95": [bootstrap[125], bootstrap[4874]],
            "by_anchor": {str(a): {"pairs": len(v), "correct": sum(v),
                                   "accuracy": sum(v) / len(v)}
                          for a, v in sorted(by_anchor.items())}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-audit", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    began = time.time()
    torch.manual_seed(11)
    torch.set_num_threads(6)
    manifest = json.loads(args.manifest.read_text())
    collection = json.loads(args.collection_audit.read_text())
    audit = json.loads(args.state_audit.read_text())
    manifest_sha = digest(args.manifest)
    if manifest["schema"] != "policy_group_relative_manifest_v1" or \
            collection["manifest_sha256"] != manifest_sha or \
            audit["manifest_sha256"] != manifest_sha or \
            audit["schema"] != "group_relative_state_cache_audit_v1":
        raise ValueError("group-four source audit mismatch")
    data = {}
    summaries = {}
    for part in ("fit", "development"):
        summaries[part], data[part] = load_part(
            manifest, part, args.turn_root, args.state_root, manifest_sha,
            audit["source_id"])
        if summaries[part]["trajectories"] != audit["parts"][part]["trajectories"]:
            raise ValueError(f"cache coverage mismatch {part}")
    fit = data["fit"]
    dev = data["development"]
    if summaries["development"]["matched_pairs"] < 100 or \
            summaries["development"]["matched_groups"] < 15:
        raise ValueError("development sample floor failed")
    fit_scenes = {x["scene_id"] for x in fit}
    dev_scenes = {x["scene_id"] for x in dev}
    if fit_scenes & dev_scenes:
        raise ValueError("scene leakage")
    # The fixed fit-only scale avoids using development outcomes to tune
    # any representation transform.
    raw_fit = torch.stack([x["difference"] for x in fit])
    raw_dev = torch.stack([x["difference"] for x in dev])
    # A common positive scale for each coordinate preserves the sign of a
    # same-group value difference while controlling hidden-state anisotropy.
    scale = raw_fit.std(dim=0, unbiased=False).clamp_min(.05)
    train = raw_fit / scale
    valid = raw_dev / scale
    train = F.normalize(train, dim=1)
    valid = F.normalize(valid, dim=1)
    if not bool(torch.isfinite(train).all() and torch.isfinite(valid).all()):
        raise ValueError("nonfinite normalized state difference")
    by_group = defaultdict(int)
    for pair in fit:
        by_group[pair["group_id"]] += 1
    weights = torch.tensor([1 / by_group[pair["group_id"]] for pair in fit])
    weights = weights / weights.mean()
    vector = torch.zeros(train.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([vector], lr=1.0, max_iter=200,
                                  line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = train @ vector
        loss = (weights * F.softplus(-logits)).mean() + L2 * vector.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        fit_scores = train @ vector
        dev_scores = valid @ vector
    if not bool(torch.isfinite(vector).all() and torch.isfinite(fit_scores).all()
                and torch.isfinite(dev_scores).all()):
        raise ValueError("nonfinite fitted value readout")
    result = {"schema": "group4_future_success_linear_development_v1",
              "interpretation": "exploratory train-scene outcome-value ranking; no navigation gain",
              "manifest_sha256": manifest_sha,
              "collection_audit_sha256": digest(args.collection_audit),
              "state_audit_sha256": digest(args.state_audit),
              "encoder_source_id": audit["source_id"],
              "settings": {"anchors": ANCHORS, "requires_later_motion_turn": True,
                           "l2": L2, "lbfgs_max_iter": 200, "seed": 11,
                           "group_inverse_frequency_weights": True},
              "coverage": summaries,
              "fit": metrics(fit, fit_scores),
              "development": metrics(dev, dev_scores),
              "elapsed_seconds": time.time() - began}
    gate = {"development_pairs_at_least_100": len(dev) >= 100,
            "development_groups_at_least_15": summaries["development"]["matched_groups"] >= 15,
            "comparison_accuracy_at_least_0_70": result["development"]["accuracy"] >= .70,
            "group_macro_accuracy_at_least_0_70":
                result["development"]["group_macro_accuracy"] >= .70}
    result["predeclared_gate"] = gate
    result["eligible_for_one_time_model_audit"] = all(gate.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    weights_path = args.output.with_suffix(".pt")
    torch.save({"schema": "group4_future_success_linear_weights_v1",
                "manifest_sha256": manifest_sha,
                "encoder_source_id": audit["source_id"],
                "scale": scale.detach(), "vector": vector.detach()}, weights_path)
    print(json.dumps({"coverage": summaries, "fit": result["fit"],
                      "development": result["development"],
                      "predeclared_gate": gate,
                      "eligible_for_one_time_model_audit": all(gate.values()),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
