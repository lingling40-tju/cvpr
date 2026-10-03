"""Fixed pilot: tune reward-encoder LoRA on group-four and prefix contrasts.

Only fit-scene histories supply gradients. The previously fitted readout
and coordinate scale are frozen; development is not opened by this CLI.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model
from fit_group4_future_success_linear import ANCHORS, FAILURES, SUCCESS


def group_id(plan: dict) -> str:
    return f"s{plan['seed']}_e{plan['episode_id']}"


def record_id(plan: dict) -> str:
    return f"{group_id(plan)}_v{plan['variant']}"


def load_group_pairs(manifest: dict, manifest_sha: str,
                     turn_root: Path) -> tuple[dict[str, list[tuple]], set[str]]:
    groups = defaultdict(list)
    scenes = set()
    for plan in manifest["selected"]["fit"]:
        rid = record_id(plan)
        path = turn_root / "fit" / "records" / f"{rid}.json"
        record = json.loads(path.read_text())
        if record["record_id"] != rid or \
                record["manifest_sha256"] != manifest_sha or \
                record["scene_id"] != plan["scene_id"] or \
                record["terminal_mode"] != plan["terminal_mode"]:
            raise ValueError(f"bad fit group record {rid}")
        last = max(t["original_turn_index"] for t in record["turns"])
        anchors = {int(t["original_turn_index"]): index
                   for index, t in enumerate(record["turns"], 1)
                   if t["original_turn_index"] in ANCHORS and
                   t["original_turn_index"] < last}
        groups[group_id(plan)].append((plan, record, anchors))
        scenes.add(plan["scene_id"])
    if len(groups) != 160 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("incomplete fit group-four rollouts")
    by_scene = defaultdict(list)
    for gid, rows in groups.items():
        if len({r[0]["scene_id"] for r in rows}) != 1:
            raise ValueError(f"mixed group scene {gid}")
        scene = rows[0][0]["scene_id"]
        for anchor in ANCHORS:
            good = [r for r in rows if r[0]["terminal_mode"] == SUCCESS and
                    anchor in r[2]]
            bad = [r for r in rows if r[0]["terminal_mode"] in FAILURES and
                   anchor in r[2]]
            for gp, gr, gi in good:
                for bp, br, bi in bad:
                    by_scene[scene].append((
                        {"record": gr, "root": turn_root / "fit"}, gi[anchor],
                        {"record": br, "root": turn_root / "fit"}, bi[anchor]))
    if sum(map(len, by_scene.values())) != 441 or \
            len({r[0]["scene_id"] for rows in groups.values() for r in rows}) != 38:
        raise ValueError("fit group comparison count or scenes changed")
    return dict(by_scene), scenes


def load_expert_pairs(manifest: dict, manifest_sha: str,
                      expert_root: Path) -> tuple[dict[str, list[tuple]], set[str]]:
    by_scene = defaultdict(list)
    scenes = set()
    for row in manifest["selected"]["fit"]:
        eid = str(row["episode_id"])
        path = expert_root / row["source_part"] / "records" / f"{eid}.json"
        if digest(path) != row["record_sha256"]:
            raise ValueError(f"expert record hash changed {eid}")
        record = json.loads(path.read_text())
        if record["episode_id"] != eid or \
                record["scene_id"] != row["scene_id"] or \
                record["trajectory_id"] != row["trajectory_id"] or \
                record["turn_count"] != row["turn_count"] or \
                record["wrong_instruction_episode_id"] != row["wrong_episode_id"]:
            raise ValueError(f"expert record changed {eid}")
        item = {"record": record,
                "root": expert_root / row["source_part"]}
        for anchor in ANCHORS:
            if anchor < row["turn_count"]:
                by_scene[row["scene_id"]].append((item, anchor))
        scenes.add(row["scene_id"])
    if len(manifest["selected"]["fit"]) != 596 or \
            sum(map(len, by_scene.values())) != 1107 or len(scenes) != 38:
        raise ValueError("fit expert prefix count or scenes changed")
    return dict(by_scene), scenes


def encode(processor, model, item: dict, count: int,
           instruction: str | None = None) -> torch.Tensor:
    inputs = build_inputs(processor, item, count, instruction).to("cuda")
    hidden = model(**inputs, output_hidden_states=False,
                   use_cache=False).logits[0, -1].float()
    if hidden.shape != (2048,) or not bool(torch.isfinite(hidden).all()):
        raise ValueError("nonfinite encoder state")
    return hidden


def loss_pair(processor, model, first: dict, first_count: int,
              second: dict, second_count: int, scale: torch.Tensor,
              vector: torch.Tensor, first_instruction: str | None = None,
              second_instruction: str | None = None) -> tuple[torch.Tensor, float]:
    a = encode(processor, model, first, first_count, first_instruction)
    b = encode(processor, model, second, second_count, second_instruction)
    margin = F.normalize((a - b) / scale, dim=0) @ vector
    if not bool(torch.isfinite(margin)):
        raise ValueError("nonfinite pair margin")
    return F.softplus(-margin), float(margin.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-collection-audit", type=Path, required=True)
    parser.add_argument("--group-state-audit", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-prefix-audit", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.set_num_threads(6)
    rng = random.Random(11)
    group = json.loads(args.group_manifest.read_text())
    collection = json.loads(args.group_collection_audit.read_text())
    state_audit = json.loads(args.group_state_audit.read_text())
    expert = json.loads(args.expert_manifest.read_text())
    prefix_audit = json.loads(args.expert_prefix_audit.read_text())
    initial = torch.load(args.initial_checkpoint, map_location="cpu",
                         weights_only=True)
    frozen = torch.load(args.frozen_weights, map_location="cpu",
                        weights_only=True)
    group_sha = digest(args.group_manifest)
    expert_sha = digest(args.expert_manifest)
    encoder_sha = digest(args.initial_checkpoint)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            collection["manifest_sha256"] != group_sha or \
            not collection["group_comparison_sample_gate"]["passed"] or \
            state_audit["manifest_sha256"] != group_sha or \
            state_audit["source_id"] != encoder_sha or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            expert["source_sha256"]["group_manifest"] != group_sha or \
            prefix_audit["schema"] != "group4_expert_prefix_state_audit_v1" or \
            prefix_audit["manifest_sha256"] != expert_sha or \
            prefix_audit["source_id"] != encoder_sha or \
            prefix_audit["parts"]["fit"]["instruction_prefix_contrasts"] != 1107 or \
            initial["schema"] != "history_grounding_lora_interim_v1" or \
            initial["model_config_sha256"] != digest(args.model / "config.json") or \
            frozen["schema"] != "group4_joint_value_weights_v1" or \
            frozen["source_sha256"]["group_manifest"] != group_sha or \
            frozen["encoder_source_id"] != encoder_sha:
        raise ValueError("frozen training source mismatch")
    group_pairs, group_scenes = load_group_pairs(
        group, group_sha, args.group_turn_root)
    expert_pairs, expert_scenes = load_expert_pairs(
        expert, expert_sha, args.expert_root)
    if group_scenes != expert_scenes:
        raise ValueError("fit scene partitions differ")
    scale = frozen["scale"].float().cuda()
    vector = frozen["vector"].float().cuda()
    if scale.shape != (2048,) or vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all()) or \
            not bool(torch.isfinite(vector).all()):
        raise ValueError("bad frozen readout")
    processor, model, _, trainable = load_model(args.model)
    set_peft_model_state_dict(model, initial["adapter"])
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    steps = 8 if args.smoke else 512
    started = time.time()
    recent = []
    for step in range(1, steps + 1):
        kind = "outcome" if step % 2 else "instruction"
        if kind == "outcome":
            scene = rng.choice(sorted(group_pairs))
            first, first_count, second, second_count = rng.choice(group_pairs[scene])
            loss, margin = loss_pair(processor, model, first, first_count,
                                     second, second_count, scale, vector)
        else:
            scene = rng.choice(sorted(expert_pairs))
            item, count = rng.choice(expert_pairs[scene])
            instruction = item["record"]["instruction"]
            wrong = item["record"]["wrong_instruction"]
            loss, margin = loss_pair(processor, model, item, count, item, count,
                                     scale, vector, instruction, wrong)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite {kind} loss at microstep {step}")
        (loss / 4).backward()
        recent.append({"kind": kind, "loss": float(loss.detach()),
                       "margin": margin})
        if step % 4 == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0:
                raise ValueError(f"invalid gradient at microstep {step}: {norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == 1:
            part = recent[-32:]
            print(json.dumps({"microstep": step,
                              "recent_loss": sum(x["loss"] for x in part) / len(part),
                              "recent_margin": sum(x["margin"] for x in part) / len(part),
                              "elapsed_seconds": time.time() - started}), flush=True)
    report = {"schema": "group4_prefix_contrast_lora_training_v1",
              "smoke": args.smoke, "seed": 11,
              "microsteps": steps, "gradient_accumulation": 4,
              "optimizer_updates": steps // 4,
              "optimizer": {"name": "AdamW", "learning_rate": 5e-5,
                            "weight_decay": .01, "gradient_clip": 1.0},
              "trainable_lora_parameters": trainable,
              "fit_group_outcome_pairs": sum(map(len, group_pairs.values())),
              "fit_expert_prefix_pairs": sum(map(len, expert_pairs.values())),
              "fit_scenes": len(group_scenes),
              "source_sha256": {
                  "group_manifest": group_sha,
                  "group_collection_audit": digest(args.group_collection_audit),
                  "group_state_audit": digest(args.group_state_audit),
                  "expert_manifest": expert_sha,
                  "expert_prefix_audit": digest(args.expert_prefix_audit),
                  "initial_checkpoint": encoder_sha,
                  "frozen_weights": digest(args.frozen_weights),
                  "model_config": digest(args.model / "config.json")},
              "selection": "fixed final adapter; no development checkpoint selection",
              "elapsed_seconds": time.time() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.smoke:
        adapter = {k: v.detach().cpu().clone() for k, v in
                   get_peft_model_state_dict(model).items()}
        checkpoint = {"schema": "group4_prefix_contrast_lora_v1",
                      "adapter": adapter, "source_sha256": report["source_sha256"],
                      "training_microsteps": steps,
                      "model_config_sha256": report["source_sha256"]["model_config"]}
        temporary = args.output / "final_adapter.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output / "final_adapter.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
