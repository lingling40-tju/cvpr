"""Fit an unbounded potential with train-fit expert instruction swaps.

The model reads policy-format RGB/action history plus an instruction. It
never reads goal coordinates or geodesic distances. Train-only simulator
distances supervise local progress, same-start rankings, and *actual*
crossed-goal sign reversals. This variant adds verified correct-versus-
wrong instruction contrasts on R2R-train expert routes to the unbounded
potential fit. The same scene-disjoint development gate decides whether
a prospective audit or online n=4 reward is warranted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import train_policy_progress_lora as prior
from history_grounding_lora import digest, load_model, load_part, signed_pairs
from train_balanced_change_lora import load_expanded_fit


RENDER_SHA = "43bf8bc8af46f07051d299810c1dd2975034a0b84da7f278b98fda5db761006c"
RGB_AUDIT_SHA = "047d0711ec4112f18eea6247538bbd2e914a248e5e2ecb8af4f23aa0c8385a45"
CROSS_SHA = "1faa89a1284da0d75c9d1b3f785bc89da41dac6abbea06e58daa3e15aa8a7137"
CROSS_AUDIT_SHA = "03eaeee0991c695dc455f71cd8577ba80f2df7d72fdd2e99acde5970c46a609d"


class UnboundedPotentialHead(torch.nn.Module):
    """Keep the original initialized head weights but return its raw value."""

    def __init__(self, bounded_head: torch.nn.Module):
        super().__init__()
        self.net = bounded_head.net

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(hidden.float())
        return scores[..., 0], scores[..., 1]


def by_scene_episode(items: list[dict]) -> dict:
    index = defaultdict(lambda: defaultdict(list))
    for item in items:
        record = item["item"]["record"] if "item" in item else item["record"]
        index[record["scene_id"]][str(record["episode_id"])].append(item)
    if not index:
        raise ValueError("empty scene/episode index")
    return index


def load_new_fit(args, fit: dict) -> tuple[list[dict], dict, dict]:
    if (digest(args.render_manifest) != RENDER_SHA
            or digest(args.rgb_audit) != RGB_AUDIT_SHA
            or digest(args.cross_manifest) != CROSS_SHA
            or digest(args.cross_audit) != CROSS_AUDIT_SHA):
        raise ValueError("new fit source checksum changed")
    render = json.loads(args.render_manifest.read_text())
    rgb_audit = json.loads(args.rgb_audit.read_text())
    cross = json.loads(args.cross_manifest.read_text())
    cross_audit = json.loads(args.cross_audit.read_text())
    if (render["schema"] != "policy_process_train_manifest_v1"
            or render["targets"] != {"fit": 512, "development": 0, "audit": 0}
            or rgb_audit["schema"] != "control_fit_render_audit_v1"
            or rgb_audit["source_sha256"]["render_manifest"] != RENDER_SHA
            or not rgb_audit["passed_sample_gate"]
            or rgb_audit["max_abs_turn_distance_drift_m"] > 1e-4
            or cross["schema"] != "cross_goal_fit_manifest_v1"
            or cross["trajectories"] != 500
            or cross_audit["schema"] != "cross_goal_fit_label_audit_v1"
            or cross_audit["source_sha256"]["manifest"] != CROSS_SHA
            or not cross_audit["sample_gate"]["passed"]
            or cross_audit["maximum_correct_distance_drift_m"] > 1e-4):
        raise ValueError("fit audit or label gate invalid")
    # Expert routes can share a train-fit episode ID with a new policy
    # rollout. They are supervision on the same fit scene, not a holdout.
    old_policy_ids = {str(item["record"]["episode_id"])
                      for item in fit["policy"]}
    by_record = {}
    new_items = []
    for plan in render["selected"]["fit"]:
        rid = f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"
        record = json.loads((args.render_root / "fit" / "records" /
                             f"{rid}.json").read_text())
        if (record["schema"] != "policy_process_turn_record_v1"
                or record["manifest_sha256"] != RENDER_SHA
                or record["record_id"] != rid
                or record["scene_id"] not in fit["scenes"]
                or record["scene_id"] != plan["scene_id"]
                or str(record["episode_id"]) in old_policy_ids):
            raise ValueError(f"new fit record/source isolation failed: {rid}")
        distances = [record["start_distance_to_goal_for_label_only"]] + [
            turn["distance_to_goal_for_label_only"] for turn in record["turns"]]
        if not all(math.isfinite(float(x)) and x >= 0 for x in distances):
            raise ValueError(f"invalid new fit labels: {rid}")
        item = {"record": record, "root": args.render_root / "fit",
                "distances": distances, "pairs": signed_pairs(distances)}
        new_items.append(item)
        by_record[rid] = item
    if len(new_items) != 512 or len({str(item["record"]["episode_id"])
                                     for item in new_items}) != 256:
        raise ValueError("new fit coverage changed")
    cross_by_record = {}
    for plan in cross["plans"]:
        rid = f"s{plan['seed']}_e{plan['episode_id']}_v{plan['variant']}"
        item = by_record.get(rid)
        label = json.loads((args.cross_root / "records" /
                            f"{rid}.json").read_text())
        if (item is None or label["schema"] != "cross_goal_fit_label_record_v1"
                or label["manifest_sha256"] != CROSS_SHA
                or label["record_id"] != rid
                or label["wrong_episode_id"] != plan["wrong_episode_id"]
                or label["correct_distance_m_for_label_only"]
                    != item["distances"]
                or len(label["wrong_distance_m_for_label_only"])
                    != len(item["distances"])):
            raise ValueError(f"crossed-goal/visual source mismatch: {rid}")
        cross_by_record[rid] = {"item": item, "label": label,
                                "wrong_instruction": plan["wrong_instruction"]}
    if len(cross_by_record) != 500:
        raise ValueError("crossed-goal coverage changed")
    return new_items, cross_by_record, cross_audit


def crossed_examples(cross_by_record: dict) -> dict[str, dict]:
    rows = {"correct_forward": [], "correct_backward": []}
    for source in cross_by_record.values():
        label = source["label"]
        correct = label["correct_distance_m_for_label_only"]
        wrong = label["wrong_distance_m_for_label_only"]
        for after in range(1, min(len(correct), 7)):
            dc = correct[after - 1] - correct[after]
            dw = wrong[after - 1] - wrong[after]
            if dc >= .5 and dw <= -.5:
                kind = "correct_forward"
            elif dc <= -.5 and dw >= .5:
                kind = "correct_backward"
            else:
                continue
            rows[kind].append({**source, "after": after,
                               "direction": 1.0 if dc > 0 else -1.0})
    if min(map(len, rows.values())) < 50:
        raise ValueError("insufficient balanced early crossed-goal labels")
    return {kind: by_scene_episode(values) for kind, values in rows.items()}


def same_start_pairs(items: list[dict]) -> dict:
    by_id = defaultdict(list)
    for item in items:
        by_id[str(item["record"]["episode_id"])].append(item)
    pairs = []
    for rows in by_id.values():
        if len(rows) != 2:
            raise ValueError("new fit episode does not have two RGB variants")
        a, b = rows
        if (a["record"]["scene_id"] != b["record"]["scene_id"]
                or a["record"]["instruction"] != b["record"]["instruction"]):
            raise ValueError("same-start group mismatch")
        choices = []
        for anchor in (3, 6):
            if len(a["record"]["turns"]) < anchor or \
                    len(b["record"]["turns"]) < anchor:
                continue
            gap = a["distances"][anchor] - b["distances"][anchor]
            if abs(gap) >= 1.0:
                choices.append((abs(gap), anchor, gap))
        if not choices:
            continue
        _, anchor, gap = max(choices)
        better, worse = (a, b) if gap < 0 else (b, a)
        pairs.append({"item": better, "worse": worse, "anchor": anchor})
    if len(pairs) < 100:
        raise ValueError("insufficient same-start ordinal fit pairs")
    return by_scene_episode(pairs)


def cross_loss(processor, model, head, row: dict) -> torch.Tensor:
    item, after = row["item"], row["after"]
    wrong_instruction = row["wrong_instruction"]
    before = after - 1
    correct_gain = (prior.progress(processor, model, head, item, after)
                    - prior.progress(processor, model, head, item, before))
    wrong_gain = (prior.progress(processor, model, head, item,
                                 after, wrong_instruction)
                  - prior.progress(processor, model, head, item,
                                   before, wrong_instruction))
    sign = row["direction"]
    return (F.softplus(.2 - sign * correct_gain)
            + F.softplus(.2 + sign * wrong_gain)
            + .5 * F.softplus(.4 - sign * (correct_gain - wrong_gain)))


def group_loss(processor, model, head, row: dict) -> torch.Tensor:
    good = prior.progress(processor, model, head,
                          row["item"], row["anchor"])
    bad = prior.progress(processor, model, head,
                         row["worse"], row["anchor"])
    return F.softplus(.25 - (good - bad))


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("scene-split", "expert-manifest", "expert-labels",
                 "expert-root", "policy-manifest", "policy-root",
                 "policy-audit", "expanded-manifest", "expanded-root",
                 "expanded-audit", "render-manifest", "render-root",
                 "rgb-audit", "cross-manifest", "cross-root",
                 "cross-audit", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.preflight:
        raise ValueError("smoke and preflight are exclusive")
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
    fit, _ = load_expanded_fit(
        common, args.expanded_manifest, args.expanded_root,
        args.expanded_audit)
    dev = None if args.smoke or args.preflight else load_part(
        "development", *common)
    new_items, cross_by_record, cross_audit = load_new_fit(args, fit)
    fit["policy"].extend(new_items)
    safe_experts = [item for item in fit["expert"] if item["safe_swap"]]
    if len(safe_experts) < 100:
        raise ValueError("insufficient safe fit expert instruction swaps")
    expert_index = prior.index_experts(safe_experts)
    by_class = {kind: prior.indexed(fit["policy"], kind)
                for kind in ("forward", "backward", "stationary")}
    crossed = crossed_examples(cross_by_record)
    grouped = same_start_pairs(new_items)
    cross_counts = {kind: sum(len(values) for episodes in index.values()
                              for values in episodes.values())
                    for kind, index in crossed.items()}
    grouped_count = sum(len(values) for episodes in grouped.values()
                        for values in episodes.values())
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {"schema": "unbounded_expert_cross_goal_potential_fit_preflight_v1",
                  "interpretation": "Fit-only source and label coverage; no model fit or navigation result",
                  "fit_policy_trajectories": len(fit["policy"]),
                  "fit_new_trajectories": len(new_items),
                  "fit_crossed_trajectories": len(cross_by_record),
                  "fit_safe_expert_swaps": len(safe_experts),
                  "crossed_early_examples": cross_counts,
                  "same_start_pairs": grouped_count,
                  "all_crossed_turns": cross_audit["crossed_turns_0p5m"],
                  "source_sha256": {"render_manifest": RENDER_SHA,
                                    "cross_manifest": CROSS_SHA,
                                    "cross_audit": CROSS_AUDIT_SHA}}
        (args.output / "preflight.json").write_text(
            json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    processor, model, bounded_head, trainable = load_model(args.model)
    head = UnboundedPotentialHead(bounded_head)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 7e-5},
        {"params": head.parameters(), "lr": 2e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    history, best = [], None
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        head.train()
        kind = ("forward", "backward", "stationary",
                "correct_forward", "correct_backward", "group",
                "expert")[(step - 1) % 7]
        if kind in by_class:
            item = prior.choose(by_class[kind], rng)
            after = rng.choice(prior.pair_indices(item)[kind])
            loss = prior.local_loss(processor, model, head, item, after, kind)
        elif kind in crossed:
            loss = cross_loss(processor, model, head,
                              prior.choose(crossed[kind], rng))
        elif kind == "expert":
            loss = prior.instruction_loss(processor, model, head,
                                          prior.choose(expert_index, rng))
        else:
            loss = group_loss(processor, model, head,
                              prior.choose(grouped, rng))
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite {kind} loss at step {step}")
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
                              "elapsed_seconds": time.time() - started}),
                  flush=True)
        if not args.smoke and step in (250, 500, 1000, 1500):
            metrics = prior.evaluate(processor, model, head, dev, small=True)
            decision = prior.gate(metrics, preliminary=True)
            history.append({"step": step, "development_small": metrics,
                            "preliminary_gate": decision})
            print(json.dumps(history[-1]), flush=True)
            quality = (sum(decision.values()),
                       metrics["balanced_direction_accuracy"],
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
                          "fit_trajectories": len(fit["policy"]),
                          "fit_safe_expert_swaps": len(safe_experts),
                          "crossed_early_examples": cross_counts,
                          "same_start_pairs": grouped_count,
                          "elapsed_seconds": time.time() - started}),
              flush=True)
        return
    if best is None:
        raise ValueError("no development-selected checkpoint")
    set_peft_model_state_dict(model, best[2])
    head.load_state_dict(best[3])
    full_dev = (prior.evaluate(processor, model, head, dev, small=False)
                if all(prior.gate(best[4], preliminary=True).values()) else None)
    sources = {"scene_split": digest(args.scene_split),
               "expert_manifest": digest(args.expert_manifest),
               "old_policy_manifest": digest(args.policy_manifest),
               "expanded_policy_manifest": digest(args.expanded_manifest),
               "render_manifest": RENDER_SHA,
               "rgb_audit": RGB_AUDIT_SHA,
               "cross_manifest": CROSS_SHA,
               "cross_audit": CROSS_AUDIT_SHA,
               "model_config": digest(args.model / "config.json")}
    report = {"schema": "unbounded_expert_cross_goal_potential_lora_development_v1",
              "interpretation": "Scene-disjoint R2R-train development only; no audit, RL, or navigation result",
              "seed": seed, "microsteps": steps,
              "gradient_accumulation": accumulation,
              "lora_parameters": trainable,
              "selected_step": best[1], "source_sha256": sources,
              "fit_policy_trajectories": len(fit["policy"]),
              "fit_new_trajectories": len(new_items),
              "fit_crossed_trajectories": len(cross_by_record),
              "fit_safe_expert_swaps": len(safe_experts),
              "fit_crossed_early_examples": cross_counts,
              "fit_same_start_pairs": grouped_count,
              "fit_all_crossed_turns": cross_audit["crossed_turns_0p5m"],
              "selected_small_development": best[4],
              "small_development_gate": prior.gate(best[4], preliminary=True),
              "full_development": full_dev,
              "full_development_gate": prior.gate(full_dev, preliminary=False)
                                       if full_dev is not None else None,
              "history": history, "elapsed_seconds": time.time() - started}
    torch.save({"schema": "unbounded_expert_cross_goal_potential_lora_v1",
                "adapter": best[2], "head": best[3],
                "selected_step": best[1], "source_sha256": sources},
               args.output / "adapter_head.pt")
    (args.output / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_step": best[1],
                      "small_gate": report["small_development_gate"],
                      "full_gate": report["full_development_gate"]}),
          flush=True)


if __name__ == "__main__":
    main()
