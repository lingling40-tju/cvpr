"""Jointly align policy outcomes and instruction-specific train goal views.

The visual score itself takes one RGB embedding and one instruction embedding.
Training compares successful versus failed trajectories from the same
group-four train episode, plus natural same-start/different-goal instruction
pairs. Scene-disjoint development/audit splits are fixed in the policy-pair
manifest. This is an offline screen; no RL policy update occurs here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_goal_contrastive import GoalMatcher
from fit_panoramic_goal import evaluate_panoramas, load_views, score, subset


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_goal_data(manifest: dict, fit_path: Path, cal_path: Path, device: str) -> dict:
    fit = load_views(fit_path, manifest, "fit", device)
    cal = load_views(cal_path, manifest, "calibration", device)
    if fit["backbone"] != cal["backbone"] or \
            fit["model_config_sha256"] != cal["model_config_sha256"]:
        raise ValueError("goal backbone mismatch")
    offset = len(fit["ids"])
    return {"ids": fit["ids"] + cal["ids"],
            "scenes": fit["scenes"] + cal["scenes"],
            "images": torch.cat((fit["images"], cal["images"])),
            "texts": torch.cat((fit["texts"], cal["texts"])),
            "pairs": fit["pairs"] + [(a + offset, b + offset, scene)
                                     for a, b, scene in cal["pairs"]],
            "backbone": fit["backbone"],
            "model_config_sha256": fit["model_config_sha256"]}


def goal_scene_subset(data: dict, scenes: set[str]) -> dict:
    active = {scene for _, _, scene in data["pairs"] if scene in scenes}
    if len(active) < len(scenes) - 2:
        raise ValueError("too few goal-pair scenes in split")
    return subset(data, active)


@torch.no_grad()
def evaluate_policy(model: GoalMatcher, data: dict, pair_indices: list[int]) -> dict:
    model.eval()
    device = data["images"].device
    success_idx = torch.tensor([2 * index for index in pair_indices],
                               dtype=torch.long, device=device)
    failure_idx = success_idx + 1
    visual, text = data["images"], data["texts"]
    success_score = score(model, visual[success_idx, -2:], text[success_idx])
    failure_score = score(model, visual[failure_idx, -2:], text[failure_idx])
    success_start = score(model, visual[success_idx, :1], text[success_idx])
    failure_start = score(model, visual[failure_idx, :1], text[failure_idx])
    raw = lambda images, texts: (images * texts[:, None, :]).sum(-1).max(-1).values
    raw_success = raw(visual[success_idx, -2:], text[success_idx])
    raw_failure = raw(visual[failure_idx, -2:], text[failure_idx])
    if any(data["scene_ids"][index] != data["scene_ids"][index + 1]
           for index in success_idx.tolist()):
        raise ValueError("policy pair scene mismatch")
    scenes = [data["scene_ids"][index] for index in success_idx.tolist()]
    hits = (success_score > failure_score).float()
    per_scene = []
    for scene in sorted(set(scenes)):
        mask = torch.tensor([row == scene for row in scenes], device=device)
        per_scene.append({"scene": scene, "pairs": int(mask.sum().item()),
                          "accuracy": hits[mask].mean().item()})
    return {"pairs": len(pair_indices), "scenes": len(per_scene),
            "success_over_failure_accuracy": hits.mean().item(),
            "raw_success_over_failure_accuracy":
                (raw_success > raw_failure).float().mean().item(),
            "success_endpoint_above_start_rate":
                (success_score > success_start).float().mean().item(),
            "failure_endpoint_above_start_rate":
                (failure_score > failure_start).float().mean().item(),
            "score_margin_mean": (success_score - failure_score).mean().item(),
            "per_scene": per_scene}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-features", type=Path, required=True)
    parser.add_argument("--goal-manifest", type=Path, required=True)
    parser.add_argument("--goal-fit-features", type=Path, required=True)
    parser.add_argument("--goal-calibration-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_manifest = json.loads(args.policy_manifest.read_text())
    policy = torch.load(args.policy_features, map_location="cpu", weights_only=False)
    if policy["schema"] != "policy_preference_features_v1" or \
            len(policy["record_ids"]) != 2 * len(policy_manifest["pairs"]) or \
            policy["manifest_sha256"] != digest(args.policy_manifest):
        raise ValueError("policy feature coverage mismatch")
    for index, pair in enumerate(policy_manifest["pairs"]):
        if policy["record_ids"][2 * index:2 * index + 2] != [
                pair["pair_id"] + "_success", pair["pair_id"] + "_failure"] or \
                policy["splits"][2 * index:2 * index + 2] != [pair["split"]] * 2:
            raise ValueError("policy pair order mismatch")
    for key in ("images", "texts"):
        policy[key] = policy[key].to(device)
    goal_manifest = json.loads(args.goal_manifest.read_text())
    goal = load_goal_data(goal_manifest, args.goal_fit_features,
                          args.goal_calibration_features, device)
    if policy["backbone"] != goal["backbone"] or \
            policy["model_config_sha256"] != goal["model_config_sha256"]:
        raise ValueError("policy/goal backbone mismatch")
    split_scenes = {key: set(value) for key, value in policy_manifest["scene_split"].items()}
    if any(split_scenes[a] & split_scenes[b] for a, b in
           (("fit", "development"), ("fit", "audit"), ("development", "audit"))):
        raise ValueError("policy scene overlap")
    goal_split = {key: goal_scene_subset(goal, split_scenes[key])
                  for key in ("fit", "development", "audit")}
    policy_indices = {key: [i for i, pair in enumerate(policy_manifest["pairs"])
                            if pair["split"] == key]
                      for key in ("fit", "development", "audit")}
    if {key: len(value) for key, value in policy_indices.items()} != \
            policy_manifest["counts"]:
        raise ValueError("policy split counts changed")
    model = GoalMatcher(policy["images"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    policy_by_scene = defaultdict(list)
    for index in policy_indices["fit"]:
        policy_by_scene[policy_manifest["pairs"][index]["scene_id"]].append(index)
    goal_by_scene = defaultdict(list)
    for a, b, scene in goal_split["fit"]["pairs"]:
        goal_by_scene[scene].append((a, b))
    policy_scenes, goal_scenes = sorted(policy_by_scene), sorted(goal_by_scene)
    best, best_step, best_state, best_dev = -float("inf"), 0, None, None
    for step in range(1, args.steps + 1):
        model.train()
        picked_policy = [rng.choice(policy_by_scene[rng.choice(policy_scenes)])
                         for _ in range(32)]
        success_idx = torch.tensor([2 * index for index in picked_policy],
                                   dtype=torch.long, device=device)
        failure_idx = success_idx + 1
        ps = score(model, policy["images"][success_idx, -2:],
                   policy["texts"][success_idx])
        pf = score(model, policy["images"][failure_idx, -2:],
                   policy["texts"][failure_idx])
        preference_loss = F.softplus(0.3 - (ps - pf)).mean()
        picked_goal = [rng.choice(goal_by_scene[rng.choice(goal_scenes)])
                       for _ in range(32)]
        pairs = torch.tensor(picked_goal, dtype=torch.long, device=device)
        a, b = pairs.T
        images, texts = goal_split["fit"]["images"], goal_split["fit"]["texts"]
        aa = score(model, images[a], texts[a])
        ab = score(model, images[a], texts[b])
        bb = score(model, images[b], texts[b])
        ba = score(model, images[b], texts[a])
        goal_loss = F.softplus(0.3 - torch.cat((aa - ab, bb - ba,
                                               aa - ba, bb - ab))).mean()
        loss = preference_loss + goal_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            dev_policy = evaluate_policy(model, policy, policy_indices["development"])
            dev_goal = evaluate_panoramas(model, goal_split["development"])
            criterion = (2 * min(dev_policy["success_over_failure_accuracy"],
                                 dev_goal["image_text_accuracy"])
                         + 0.2 * dev_goal["text_image_accuracy"])
            if criterion > best:
                best, best_step = criterion, step
                best_state = copy.deepcopy({key: value.detach().cpu()
                                            for key, value in model.state_dict().items()})
                best_dev = {"policy": dev_policy, "goal": dev_goal}
            if step - best_step >= 200:
                break
    if best_state is None:
        raise RuntimeError("no selected checkpoint")
    model.load_state_dict(best_state)
    audit_policy = evaluate_policy(model, policy, policy_indices["audit"])
    audit_goal = evaluate_panoramas(model, goal_split["audit"])
    gate = {"policy_preference_at_least_0_75":
                audit_policy["success_over_failure_accuracy"] >= .75,
            "goal_instruction_at_least_0_75": audit_goal["image_text_accuracy"] >= .75,
            "policy_above_raw_by_0_05":
                audit_policy["success_over_failure_accuracy"] -
                audit_policy["raw_success_over_failure_accuracy"] >= .05,
            "goal_above_raw_by_0_05":
                audit_goal["image_text_accuracy"] -
                audit_goal["raw_image_text_accuracy"] >= .05,
            "success_endpoint_above_start_at_least_0_65":
                audit_policy["success_endpoint_above_start_rate"] >= .65,
            "success_failure_progress_gap_at_least_0_10":
                audit_policy["success_endpoint_above_start_rate"] -
                audit_policy["failure_endpoint_above_start_rate"] >= .10}
    gate["status"] = "passed" if all(gate.values()) else "failed"
    report = {"interpretation": "Train-scene joint policy/goal visual screen; no RL result.",
              "seed": args.seed, "best_step": best_step,
              "fit_policy_pairs": len(policy_indices["fit"]),
              "fit_goal_pairs": len(goal_split["fit"]["pairs"]),
              "development_scenes": sorted(split_scenes["development"]),
              "audit_scenes": sorted(split_scenes["audit"]),
              "development": best_dev, "audit_policy": audit_policy,
              "audit_goal": audit_goal, "candidate_gate": gate}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": best_state, "feature_dim": policy["images"].shape[-1],
                "seed": args.seed, "best_step": best_step},
               args.output_dir / "joint_head.pt")
    print(json.dumps({"audit_policy": audit_policy, "audit_goal": audit_goal,
                      "candidate_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
