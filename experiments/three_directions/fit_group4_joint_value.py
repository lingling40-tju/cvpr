"""Fit one fixed linear readout for group-four outcome and instruction.

The frozen encoder produces both policy-history and expert-history states.
Only fit scenes update the readout; development checks the two objectives
separately. No val-unseen or newly collected simulator state is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest, load_part, metrics


L2 = .01


def expert_data(manifest: dict, part: str, cache_root: Path,
                manifest_sha: str, source_id: str) -> tuple[list[dict], torch.Tensor]:
    rows = manifest["selected"][part]
    values = []
    for row in rows:
        eid = str(row["episode_id"])
        item = torch.load(cache_root / part / "records" / f"{eid}.pt",
                          map_location="cpu", weights_only=True)
        if item["schema"] != "group4_joint_expert_state_v1" or \
                item["episode_id"] != eid or item["part"] != part or \
                item["manifest_sha256"] != manifest_sha or \
                item["record_sha256"] != row["record_sha256"] or \
                item["source_id"] != source_id:
            raise ValueError(f"expert state provenance mismatch {part}/{eid}")
        values.append(item["correct"].float() - item["wrong"].float())
    return rows, torch.stack(values)


def expert_metrics(rows: list[dict], scores: torch.Tensor) -> dict:
    correct = [int(value > 0) for value in scores.tolist()]
    by_scene = defaultdict(list)
    for row, label in zip(rows, correct):
        by_scene[row["scene_id"]].append(label)
    rng = random.Random(11)
    clusters = list(by_scene.values())
    samples = []
    for _ in range(5000):
        selected = [rng.choice(clusters) for _ in clusters]
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    return {"pairs": len(rows), "correct": sum(correct),
            "accuracy": sum(correct) / len(rows),
            "scenes": len(by_scene),
            "scene_macro_accuracy": sum(sum(x) / len(x) for x in by_scene.values()) /
                                    len(by_scene),
            "scene_cluster_bootstrap_95": [samples[125], samples[4874]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-collection-audit", type=Path, required=True)
    parser.add_argument("--group-state-audit", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--group-state-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-state-audit", type=Path, required=True)
    parser.add_argument("--expert-state-root", type=Path, required=True)
    parser.add_argument("--old-weights", type=Path, required=True)
    parser.add_argument("--old-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(11)
    torch.set_num_threads(6)
    group_manifest = json.loads(args.group_manifest.read_text())
    expert_manifest = json.loads(args.expert_manifest.read_text())
    group_audit = json.loads(args.group_state_audit.read_text())
    expert_audit = json.loads(args.expert_state_audit.read_text())
    old_development = json.loads(args.old_development.read_text())
    old = torch.load(args.old_weights, map_location="cpu", weights_only=True)
    group_sha = digest(args.group_manifest)
    expert_sha = digest(args.expert_manifest)
    if group_manifest["schema"] != "policy_group_relative_manifest_v1" or \
            expert_manifest["schema"] != "group4_joint_value_expert_manifest_v1" or \
            expert_manifest["source_sha256"]["group_manifest"] != group_sha or \
            group_audit["manifest_sha256"] != group_sha or \
            expert_audit["manifest_sha256"] != expert_sha or \
            group_audit["source_id"] != expert_audit["source_id"] or \
            old["schema"] != "group4_future_success_linear_weights_v1" or \
            old["manifest_sha256"] != group_sha or \
            old["encoder_source_id"] != group_audit["source_id"] or \
            old_development["schema"] != "group4_future_success_linear_development_v1" or \
            old_development["manifest_sha256"] != group_sha:
        raise ValueError("joint source/encoder mismatch")
    collection = json.loads(args.group_collection_audit.read_text())
    if collection["manifest_sha256"] != group_sha:
        raise ValueError("group collection audit mismatch")
    parts = {}
    for part in ("fit", "development"):
        coverage, outcome = load_part(group_manifest, part, args.group_turn_root,
                                      args.group_state_root, group_sha,
                                      group_audit["source_id"])
        expert_rows, expert_difference = expert_data(
            expert_manifest, part, args.expert_state_root,
            expert_sha, expert_audit["source_id"])
        if coverage["trajectories"] != group_audit["parts"][part]["trajectories"] or \
                len(expert_rows) != expert_audit["parts"][part]["expert_contrasts"]:
            raise ValueError(f"incomplete joint feature coverage {part}")
        # Outcome supervision exists only in mixed-success four-rollout
        # groups. Compare the full source partition, not just its labeled
        # subset, against the expert scene partition.
        if {row["scene_id"] for row in group_manifest["selected"][part]} != \
                {row["scene_id"] for row in expert_rows}:
            raise ValueError(f"outcome/expert scene partition mismatch {part}")
        parts[part] = (coverage, outcome, expert_rows, expert_difference)
    fit_scenes = {row["scene_id"] for row in parts["fit"][1]}
    dev_scenes = {row["scene_id"] for row in parts["development"][1]}
    if fit_scenes & dev_scenes:
        raise ValueError("fit/development scene leakage")
    scale = old["scale"].float()
    old_vector = old["vector"].float()
    if scale.shape != (2048,) or old_vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all() and torch.isfinite(old_vector).all()):
        raise ValueError("bad fit-only coordinate scale")
    matrices = {}
    for part, (_, outcome, _, expert_difference) in parts.items():
        outcome_x = F.normalize(torch.stack([x["difference"] for x in outcome]) /
                                scale, dim=1)
        expert_x = F.normalize(expert_difference / scale, dim=1)
        if not bool(torch.isfinite(outcome_x).all() and torch.isfinite(expert_x).all()):
            raise ValueError(f"nonfinite joint features {part}")
        matrices[part] = (outcome_x, expert_x)
    old_outcome_dev = metrics(parts["development"][1],
                              matrices["development"][0] @ old_vector)
    if old_outcome_dev["correct"] != old_development["development"]["correct"] or \
            old_outcome_dev["groups"] != old_development["development"]["groups"]:
        raise ValueError("old outcome control cannot be reproduced")
    old_expert_dev = expert_metrics(parts["development"][2],
                                    matrices["development"][1] @ old_vector)
    group_count = defaultdict(int)
    for pair in parts["fit"][1]:
        group_count[pair["group_id"]] += 1
    outcome_weights = torch.tensor([
        1 / group_count[pair["group_id"]] for pair in parts["fit"][1]])
    outcome_weights /= outcome_weights.mean()
    scene_count = defaultdict(int)
    for row in parts["fit"][2]:
        scene_count[row["scene_id"]] += 1
    expert_weights = torch.tensor([
        1 / scene_count[row["scene_id"]] for row in parts["fit"][2]])
    expert_weights /= expert_weights.mean()
    vector = torch.zeros(2048, requires_grad=True)
    optimizer = torch.optim.LBFGS([vector], lr=1.0, max_iter=200,
                                  line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        outcome_scores = matrices["fit"][0] @ vector
        expert_scores = matrices["fit"][1] @ vector
        loss = (outcome_weights * F.softplus(-outcome_scores)).mean() + \
               (expert_weights * F.softplus(-expert_scores)).mean() + \
               L2 * vector.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        fit_outcome_scores = matrices["fit"][0] @ vector
        fit_expert_scores = matrices["fit"][1] @ vector
        dev_outcome_scores = matrices["development"][0] @ vector
        dev_expert_scores = matrices["development"][1] @ vector
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("nonfinite joint readout")
    fit_outcome = metrics(parts["fit"][1], fit_outcome_scores)
    fit_expert = expert_metrics(parts["fit"][2], fit_expert_scores)
    dev_outcome = metrics(parts["development"][1], dev_outcome_scores)
    dev_expert = expert_metrics(parts["development"][2], dev_expert_scores)
    floor = {"outcome_pairs_at_least_100": dev_outcome["pairs"] >= 100,
             "outcome_groups_at_least_15": dev_outcome["groups"] >= 15,
             "expert_pairs_at_least_100": dev_expert["pairs"] >= 100,
             "expert_scenes_at_least_8": dev_expert["scenes"] >= 8}
    gate = {**floor,
            "outcome_accuracy_at_least_0_70": dev_outcome["accuracy"] >= .70,
            "outcome_group_macro_at_least_0_70":
                dev_outcome["group_macro_accuracy"] >= .70,
            "expert_accuracy_at_least_0_75": dev_expert["accuracy"] >= .75,
            "expert_scene_macro_at_least_0_70":
                dev_expert["scene_macro_accuracy"] >= .70}
    result = {"schema": "group4_joint_value_development_v1",
              "interpretation": "exploratory train-scene readout; no navigation gain",
              "source_sha256": {
                  "group_manifest": group_sha,
                  "group_collection_audit": digest(args.group_collection_audit),
                  "group_state_audit": digest(args.group_state_audit),
                  "expert_manifest": expert_sha,
                  "expert_state_audit": digest(args.expert_state_audit),
                  "old_weights": digest(args.old_weights)},
              "encoder_source_id": group_audit["source_id"],
              "optimizer": {"outcome_task_weight": 1.0,
                            "expert_task_weight": 1.0,
                            "l2": L2, "max_lbfgs_iterations": 200,
                            "seed": 11,
                            "outcome_inverse_group_weight": True,
                            "expert_inverse_scene_weight": True},
              "fit": {"outcome": fit_outcome, "expert": fit_expert},
              "development": {"outcome": dev_outcome,
                              "expert": dev_expert,
                              "old_outcome_only_readout": {
                                  "outcome": old_outcome_dev,
                                  "expert": old_expert_dev}},
              "predeclared_gate": gate,
              "eligible_for_model_audit": all(gate.values()),
              "elapsed_seconds": time.time() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    weights_path = args.output.with_suffix(".pt")
    torch.save({"schema": "group4_joint_value_weights_v1",
                "source_sha256": result["source_sha256"],
                "encoder_source_id": group_audit["source_id"],
                "scale": scale, "vector": vector.detach()}, weights_path)
    print(json.dumps({"fit": result["fit"],
                      "development": result["development"],
                      "predeclared_gate": gate,
                      "eligible_for_model_audit": all(gate.values()),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
