"""Learn group-relative visual route progress from same-start train pairs.

The model sees only RGB/action history and instruction. Simulator distance
constructs fit/development pair labels and never enters its prompt. This is
an offline representation gate, not a navigation reward result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import itertools
import json
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import digest, load_model, load_part
from train_balanced_change_lora import load_expanded_fit
import train_policy_progress_lora as prior
from train_unbounded_expert_cross_goal_potential_lora import (
    UnboundedPotentialHead, load_new_fit,
)


BASELINE_EPISODE_MACRO = {3: 0.6827925925925926,
                          6: 0.6750902739820072}


def action_prefix(item: dict, anchor: int) -> tuple[str, ...]:
    return tuple(action for turn in item["record"]["turns"][:anchor]
                 for action in turn["motion_actions"])


def forward_meters(item: dict, anchor: int) -> float:
    centimeters = 0
    for action in action_prefix(item, anchor):
        if action.startswith("move forward"):
            match = re.fullmatch(r"move forward (\d+)cm", action)
            if match is None:
                raise ValueError(f"unknown forward action: {action}")
            centimeters += int(match.group(1))
    return centimeters / 100.0


def build_pairs(items: list[dict]) -> tuple[dict[int, list[dict]], dict]:
    grouped = defaultdict(list)
    for item in items:
        record = item["record"]
        grouped[(record["scene_id"], str(record["episode_id"]),
                 record["instruction"])].append(item)
    pairs: dict[int, list[dict]] = {3: [], 6: []}
    counts = {}
    for anchor in (3, 6):
        for (scene, eid, _), routes in grouped.items():
            available = [item for item in routes
                         if len(item["distances"]) > anchor]
            for left, right in itertools.combinations(available, 2):
                gap = float(left["distances"][anchor]
                            - right["distances"][anchor])
                if abs(gap) < 1.0:
                    continue
                if action_prefix(left, anchor) == action_prefix(right, anchor):
                    raise ValueError("distance gap despite identical action prefix")
                commanded = forward_meters(left, anchor) - forward_meters(right, anchor)
                hard = commanded != 0 and (commanded > 0) == (gap > 0)
                pairs[anchor].append({"left": left, "right": right,
                                      "sign": -1.0 if gap > 0 else 1.0,
                                      "hard": hard, "scene": scene,
                                      "episode_id": eid, "anchor": anchor})
        counts[str(anchor)] = {
            "pairs": len(pairs[anchor]),
            "hard_pairs": sum(row["hard"] for row in pairs[anchor]),
            "episode_ids": len({row["episode_id"] for row in pairs[anchor]}),
            "hard_episode_ids": len({row["episode_id"] for row in pairs[anchor]
                                     if row["hard"]}),
        }
    return pairs, counts


def pair_index(rows: list[dict]) -> dict:
    index = defaultdict(lambda: defaultdict(list))
    for row in rows:
        index[row["scene"]][row["episode_id"]].append(row)
    if not index:
        raise ValueError("empty pair index")
    return index


def pair_loss(processor, model, head, row: dict) -> torch.Tensor:
    anchor = row["anchor"]
    left = prior.progress(processor, model, head, row["left"], anchor)
    right = prior.progress(processor, model, head, row["right"], anchor)
    return (F.softplus(.25 - row["sign"] * (left - right))
            + .001 * (left.square() + right.square()))


def evaluate(processor, model, head, pairs: dict, experts: list[dict]) -> dict:
    model.eval()
    head.eval()
    result = {"anchors": {}}
    with torch.inference_mode():
        for anchor in (3, 6):
            scores = {}
            for row in pairs[anchor]:
                for item in (row["left"], row["right"]):
                    rid = item["record"]["record_id"]
                    if rid not in scores:
                        scores[rid] = float(prior.progress(
                            processor, model, head, item, anchor))
            episode_scores = defaultdict(list)
            hard_episode_scores = defaultdict(list)
            for row in pairs[anchor]:
                left = scores[row["left"]["record"]["record_id"]]
                right = scores[row["right"]["record"]["record_id"]]
                signed = row["sign"] * (left - right)
                point = 1.0 if signed > 0 else (0.5 if signed == 0 else 0.0)
                key = (row["scene"], row["episode_id"])
                episode_scores[key].append(point)
                if row["hard"]:
                    hard_episode_scores[key].append(point)
            episode_means = {key: sum(values) / len(values)
                             for key, values in episode_scores.items()}
            scene_scores = defaultdict(list)
            for (scene, _), value in episode_means.items():
                scene_scores[scene].append(value)
            hard_means = [sum(values) / len(values)
                          for values in hard_episode_scores.values()]
            result["anchors"][str(anchor)] = {
                "pairs": len(pairs[anchor]),
                "episode_groups": len(episode_means),
                "scenes": len(scene_scores),
                "pair_accuracy": sum(sum(v) for v in episode_scores.values())
                / len(pairs[anchor]),
                "episode_macro_accuracy": sum(episode_means.values())
                / len(episode_means),
                "scene_macro_accuracy": sum(sum(values) / len(values)
                                             for values in scene_scores.values())
                / len(scene_scores),
                "hard_episode_macro_accuracy": sum(hard_means) / len(hard_means),
                "hard_episode_groups": len(hard_means),
            }
        safe = prior.fixed_subset(
            [item for item in experts if item["safe_swap"]],
            48, "progress-expert-dev")
        correct = 0
        for item in safe:
            record = item["record"]
            end = len(record["turns"])
            right = (float(prior.progress(processor, model, head, item, end))
                     - float(prior.progress(processor, model, head, item, 0)))
            wrong_instruction = record["wrong_instruction"].strip()
            wrong = (float(prior.progress(processor, model, head, item, end,
                                          wrong_instruction))
                     - float(prior.progress(processor, model, head, item, 0,
                                            wrong_instruction)))
            correct += right > wrong
        result["instruction_gain_preference"] = correct / len(safe)
        result["instruction_pairs"] = len(safe)
    return result


def gate(metrics: dict) -> dict[str, bool]:
    result = {"instruction_grounding":
              metrics["instruction_gain_preference"] >= .75}
    for anchor in (3, 6):
        score = metrics["anchors"][str(anchor)]
        result[f"anchor{anchor}_episode_macro"] = (
            score["episode_macro_accuracy"] >= .75)
        result[f"anchor{anchor}_over_action_baseline"] = (
            score["episode_macro_accuracy"] - BASELINE_EPISODE_MACRO[anchor]
            >= .05)
        result[f"anchor{anchor}_scene_macro"] = (
            score["scene_macro_accuracy"] >= .70)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "expanded-manifest", "expanded-root",
                 "expanded-audit", "render-manifest", "render-root",
                 "rgb-audit", "cross-manifest", "cross-root",
                 "cross-audit", "model", "pair-preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.smoke:
        raise ValueError("preflight and smoke are exclusive")
    seed, accumulation = 11, 6
    steps = 6 if args.smoke else 1500
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    rng = random.Random(seed)
    common = (args.scene_split, args.expert_manifest, args.expert_labels,
              args.expert_root, args.policy_manifest, args.policy_root,
              args.policy_audit)
    fit, _ = load_expanded_fit(common, args.expanded_manifest,
                               args.expanded_root, args.expanded_audit)
    development = None if args.smoke else load_part(
        "development", *common)
    new_items, _, _ = load_new_fit(args, fit)
    fit["policy"].extend(new_items)
    fit_pairs, fit_counts = build_pairs(fit["policy"])
    dev_pairs, dev_counts = (build_pairs(development["policy"])
                             if development is not None else ({}, {}))
    preflight = json.loads(args.pair_preflight.read_text())
    if preflight["schema"] != "same_start_pairwise_progress_preflight_v1":
        raise ValueError("pair preflight schema changed")
    if preflight["source_sha256"] != {
            "old_manifest": digest(args.expanded_manifest),
            "new_manifest": digest(args.render_manifest),
            "policy_manifest": digest(args.policy_manifest)}:
        raise ValueError("pair preflight source hashes changed")
    for anchor in (3, 6):
        baseline = preflight["development"]["anchors"][str(anchor)][
            "forward_distance_only_baseline"]["episode_macro_accuracy"]
        if abs(baseline - BASELINE_EPISODE_MACRO[anchor]) > 1e-12:
            raise ValueError("action-only comparison baseline changed")
    for anchor in (3, 6):
        expected = preflight["fit"]["anchors"][str(anchor)]
        if (fit_counts[str(anchor)]["pairs"] != expected["pairs_gap_at_least_1m"]
                or fit_counts[str(anchor)]["hard_pairs"] != expected[
                    "forward_distance_only_baseline"][
                    "hard_pairs_wrong_despite_more_forward_motion"]):
            raise ValueError("fit pair source differs from audited preflight")
        if development is not None:
            expected_dev = preflight["development"]["anchors"][str(anchor)]
            if (dev_counts[str(anchor)]["pairs"]
                    != expected_dev["pairs_gap_at_least_1m"]
                    or dev_counts[str(anchor)]["hard_pairs"]
                    != expected_dev["forward_distance_only_baseline"][
                        "hard_pairs_wrong_despite_more_forward_motion"]):
                raise ValueError("development pair source changed")
    sources = {"scene_split": digest(args.scene_split),
               "expert_manifest": digest(args.expert_manifest),
               "old_policy_manifest": digest(args.policy_manifest),
               "expanded_policy_manifest": digest(args.expanded_manifest),
               "render_manifest": digest(args.render_manifest),
               "rgb_audit": digest(args.rgb_audit),
               "cross_manifest": digest(args.cross_manifest),
               "cross_audit": digest(args.cross_audit),
               "pair_preflight": digest(args.pair_preflight),
               "model_config": digest(args.model / "config.json")}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        result = {"schema": "same_start_relative_fit_preflight_v1",
                  "interpretation": "Fit/development label counts only; no model or reward result",
                  "fit_counts": fit_counts, "development_counts": dev_counts,
                  "source_sha256": sources}
        (args.output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    if args.init_checkpoint is None:
        raise ValueError("warm-start adapter path is required")
    adapter = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    prior_sources = adapter["source_sha256"]
    shared = {"scene_split": sources["scene_split"],
              "expert_manifest": sources["expert_manifest"],
              "old_policy_manifest": sources["old_policy_manifest"],
              "expanded_policy_manifest": sources["expanded_policy_manifest"],
              "render_manifest": sources["render_manifest"],
              "rgb_audit": sources["rgb_audit"],
              "cross_manifest": sources["cross_manifest"],
              "cross_audit": sources["cross_audit"],
              "model_config": sources["model_config"]}
    if (adapter["schema"] != "unbounded_expert_cross_goal_potential_lora_v1"
            or prior_sources != shared):
        raise ValueError("warm-start adapter source mismatch")
    processor, model, bounded_head, trainable = load_model(args.model)
    head = UnboundedPotentialHead(bounded_head)
    set_peft_model_state_dict(model, adapter["adapter"])
    head.load_state_dict(adapter["head"])
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 5e-5},
        {"params": head.parameters(), "lr": 1.5e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    by_anchor = {anchor: pair_index(fit_pairs[anchor]) for anchor in (3, 6)}
    hard_by_anchor = {anchor: pair_index([row for row in fit_pairs[anchor]
                                          if row["hard"]]) for anchor in (3, 6)}
    expert_index = prior.index_experts(
        [item for item in fit["expert"] if item["safe_swap"]])
    stationary_index = prior.indexed(fit["policy"], "stationary")
    history, best = [], None
    started = time.time()
    for step in range(1, steps + 1):
        model.train()
        head.train()
        kind = ("pair3", "pair6", "hard3", "hard6", "expert",
                "stationary")[(step - 1) % 6]
        if kind.startswith("pair") or kind.startswith("hard"):
            anchor = int(kind[-1])
            index = hard_by_anchor[anchor] if kind.startswith("hard") else by_anchor[anchor]
            loss = pair_loss(processor, model, head, prior.choose(index, rng))
        elif kind == "expert":
            loss = prior.instruction_loss(
                processor, model, head, prior.choose(expert_index, rng))
        else:
            item = prior.choose(stationary_index, rng)
            after = rng.choice(prior.pair_indices(item)["stationary"])
            loss = prior.local_loss(processor, model, head,
                                    item, after, "stationary")
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}: {kind}")
        (loss / accumulation).backward()
        if step % accumulation == 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(head.parameters()), 1.0)
            if not torch.isfinite(gradient) or float(gradient) <= 0:
                raise ValueError(f"invalid gradient at step {step}: {gradient}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "kind": kind,
                              "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}), flush=True)
        if not args.smoke and step in (250, 500, 1000, 1500):
            metrics = evaluate(processor, model, head,
                               dev_pairs, development["expert"])
            decision = gate(metrics)
            history.append({"step": step, "development": metrics,
                            "gate": decision})
            print(json.dumps(history[-1]), flush=True)
            quality = (sum(decision.values()),
                       sum(metrics["anchors"][str(a)]["episode_macro_accuracy"]
                           for a in (3, 6)) / 2,
                       metrics["instruction_gain_preference"], -step)
            if best is None or quality > best[0]:
                best = (quality, step,
                        {k: v.detach().cpu().clone() for k, v in
                         get_peft_model_state_dict(model).items()},
                        copy.deepcopy({k: v.detach().cpu() for k, v in
                                       head.state_dict().items()}), metrics)
    if args.smoke:
        print(json.dumps({"smoke_updates": 1,
                          "lora_parameters": trainable,
                          "fit_counts": fit_counts,
                          "elapsed_seconds": time.time() - started}), flush=True)
        return
    if best is None:
        raise ValueError("no development checkpoint")
    result = {"schema": "same_start_relative_lora_development_v1",
              "interpretation": "Reused R2R-train development only; no prospective audit, online RL, or val-unseen result",
              "seed": seed, "microsteps": steps,
              "gradient_accumulation": accumulation,
              "lora_parameters": trainable,
              "selected_step": best[1],
              "fit_counts": fit_counts,
              "development_counts": dev_counts,
              "source_sha256": sources,
              "warm_start_sha256": digest(args.init_checkpoint),
              "selected_development": best[4],
              "development_gate": gate(best[4]),
              "history": history,
              "elapsed_seconds": time.time() - started}
    torch.save({"schema": "same_start_relative_lora_v1",
                "adapter": best[2], "head": best[3],
                "selected_step": best[1], "source_sha256": sources,
                "warm_start_sha256": result["warm_start_sha256"]},
               args.output / "adapter_head.pt")
    (args.output / "development.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_step": best[1],
                      "development_gate": result["development_gate"]}),
          flush=True)


if __name__ == "__main__":
    main()
