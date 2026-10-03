"""Group-four outcome and evidence-onset branch contrast LoRA pilot.

The label-only manifest and all gates are frozen before this training run.
Only fit-scene histories supply gradients. The joint readout stays frozen.
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

from history_grounding_lora import digest, load_model
from train_group4_crossed_prefix_lora import encode, load_group_pairs, loss_pair


def load_evidence_pairs(manifest: dict, expert_root: Path) -> tuple[dict, dict]:
    by_scene = defaultdict(list)
    onset_by_scene = defaultdict(list)
    for row in manifest["selected"]["fit"]:
        a, b, anchor = row["episode_a"], row["episode_b"], row["anchor"]
        ra_root = expert_root / row["source_part_a"]
        rb_root = expert_root / row["source_part_b"]
        pa = ra_root / "records" / f"{a}.json"
        pb = rb_root / "records" / f"{b}.json"
        if digest(pa) != row["record_a_sha256"] or \
                digest(pb) != row["record_b_sha256"]:
            raise ValueError(f"expert source changed {a}/{b}")
        ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
        if ra["episode_id"] != a or rb["episode_id"] != b or \
                ra["scene_id"] != row["scene_id"] or \
                rb["scene_id"] != row["scene_id"] or \
                anchor not in (3, 6) or \
                ra["turn_count"] <= anchor or rb["turn_count"] <= anchor or \
                ra["instruction"].strip() == rb["instruction"].strip() or \
                digest(ra_root / ra["initial_image"]) != row["initial_frame_sha256"] or \
                digest(rb_root / rb["initial_image"]) != row["initial_frame_sha256"] or \
                digest(ra_root / ra["turns"][anchor - 1]["image"]) != row["frame_a_sha256"] or \
                digest(rb_root / rb["turns"][anchor - 1]["image"]) != row["frame_b_sha256"] or \
                row["frame_a_sha256"] == row["frame_b_sha256"]:
            raise ValueError(f"evidence branch changed {a}/{b}")
        if row["evidence_onset_eligible"] and \
                (anchor != 6 or not row["shared_rgb_through_3"] or
                 any(digest(ra_root / ra["turns"][i]["image"]) !=
                     digest(rb_root / rb["turns"][i]["image"]) for i in range(3))):
            raise ValueError(f"onset source changed {a}/{b}")
        pair = ({"record": ra, "root": ra_root},
                {"record": rb, "root": rb_root}, anchor)
        by_scene[row["scene_id"]].append(pair)
        if row["evidence_onset_eligible"]:
            onset_by_scene[row["scene_id"]].append(pair)
    inventory = manifest["inventory"]["fit"]
    if sum(map(len, by_scene.values())) != inventory["crossed_pairs"] or \
            sum(map(len, onset_by_scene.values())) != inventory["evidence_onset_eligible"] or \
            len(by_scene) != inventory["scenes"] or \
            inventory["crossed_pairs"] != 295 or \
            inventory["evidence_onset_eligible"] != 73 or \
            len(by_scene) != 35:
        raise ValueError("evidence fit inventory changed")
    return dict(by_scene), dict(onset_by_scene)


def margin(good: torch.Tensor, bad: torch.Tensor,
           scale: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return F.normalize((good - bad) / scale, dim=0) @ vector


def crossed_loss(processor, model, first: dict, second: dict, anchor: int,
                 scale: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    ia, ib = first["record"]["instruction"], second["record"]["instruction"]
    aa = encode(processor, model, first, anchor, ia)
    ab = encode(processor, model, first, anchor, ib)
    bb = encode(processor, model, second, anchor, ib)
    ba = encode(processor, model, second, anchor, ia)
    values = torch.stack([margin(a, b, scale, vector) for a, b in
                          ((aa, ab), (bb, ba), (aa, ba), (bb, ab))])
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite crossed margins")
    return .5 * F.softplus(-values).sum(), float(values.detach().mean())


def onset_loss(processor, model, item: dict, correct: str, wrong: str,
               scale: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    c3 = encode(processor, model, item, 3, correct)
    w3 = encode(processor, model, item, 3, wrong)
    c6 = encode(processor, model, item, 6, correct)
    w6 = encode(processor, model, item, 6, wrong)
    early = margin(c3, w3, scale, vector)
    late = margin(c6, w6, scale, vector)
    if not bool(torch.isfinite(early) and torch.isfinite(late)):
        raise ValueError("nonfinite onset margins")
    loss = 2 * (F.softplus(-(late - early)) + .25 * early.square() +
                .5 * F.softplus(-late))
    return loss, float((late - early).detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-collection-audit", type=Path, required=True)
    parser.add_argument("--group-state-audit", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
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
    evidence = json.loads(args.evidence_manifest.read_text())
    prefix_audit = json.loads(args.expert_prefix_audit.read_text())
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    frozen = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    group_sha, expert_sha = digest(args.group_manifest), digest(args.expert_manifest)
    evidence_sha, encoder_sha = digest(args.evidence_manifest), digest(args.initial_checkpoint)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            collection["manifest_sha256"] != group_sha or \
            not collection["group_comparison_sample_gate"]["passed"] or \
            state_audit["manifest_sha256"] != group_sha or \
            state_audit["source_id"] != encoder_sha or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            expert["source_sha256"]["group_manifest"] != group_sha or \
            evidence["schema"] != "group4_evidence_onset_manifest_v1" or \
            evidence["source_expert_manifest_sha256"] != expert_sha or \
            evidence_sha != "41b465132819ae6060d06b3cf0a6ff964d9be83d3ace265ae39414aa1258f26f" or \
            prefix_audit["schema"] != "group4_expert_prefix_state_audit_v1" or \
            prefix_audit["manifest_sha256"] != expert_sha or \
            prefix_audit["source_id"] != encoder_sha or \
            initial["schema"] != "history_grounding_lora_interim_v1" or \
            initial["model_config_sha256"] != digest(args.model / "config.json") or \
            frozen["schema"] != "group4_joint_value_weights_v1" or \
            frozen["source_sha256"]["group_manifest"] != group_sha or \
            frozen["encoder_source_id"] != encoder_sha:
        raise ValueError("frozen source mismatch")
    group_pairs, group_scenes = load_group_pairs(group, group_sha, args.group_turn_root)
    evidence_pairs, onset_pairs = load_evidence_pairs(evidence, args.expert_root)
    if set(evidence_pairs) - group_scenes:
        raise ValueError("expert fit scene outside policy fit scenes")
    scale, vector = frozen["scale"].float().cuda(), frozen["vector"].float().cuda()
    if scale.shape != vector.shape or scale.shape != (2048,):
        raise ValueError("invalid frozen readout")
    processor, model, _, trainable = load_model(args.model)
    set_peft_model_state_dict(model, initial["adapter"])
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    steps = 9 if args.smoke else 384
    started = time.time()
    recent = []
    for step in range(1, steps + 1):
        if step % 3:
            kind = "outcome"
            scene = rng.choice(sorted(group_pairs))
            first, first_count, second, second_count = rng.choice(group_pairs[scene])
            loss, value = loss_pair(processor, model, first, first_count,
                                    second, second_count, scale, vector)
        elif (step // 3) % 2:
            kind = "crossed"
            scene = rng.choice(sorted(evidence_pairs))
            first, second, anchor = rng.choice(evidence_pairs[scene])
            loss, value = crossed_loss(processor, model, first, second,
                                       anchor, scale, vector)
        else:
            kind = "onset"
            scene = rng.choice(sorted(onset_pairs))
            first, second, anchor = rng.choice(onset_pairs[scene])
            if anchor != 6:
                raise ValueError("onset requires six-turn divergence")
            item, other = (first, second) if rng.randrange(2) == 0 else (second, first)
            loss, value = onset_loss(processor, model, item,
                                     item["record"]["instruction"],
                                     other["record"]["instruction"],
                                     scale, vector)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite {kind} loss at {step}")
        (loss / 4).backward()
        recent.append({"kind": kind, "loss": float(loss.detach()), "margin": value})
        if step % 3 == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0:
                raise ValueError(f"invalid gradient at {step}: {norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == 1:
            window = recent[-32:]
            print(json.dumps({"microstep": step,
                              "recent_loss": sum(x["loss"] for x in window) / len(window),
                              "recent_margin": sum(x["margin"] for x in window) / len(window),
                              "elapsed_seconds": time.time() - started}), flush=True)
    report = {"schema": "group4_evidence_onset_lora_training_v1",
              "smoke": args.smoke, "seed": 11, "microsteps": steps,
              "gradient_accumulation": 3, "optimizer_updates": steps // 3,
              "model_forward_count": (steps // 3) * 8,
              "outcome_examples": (steps // 3) * 2,
              "crossed_examples": (steps // 3 + 1) // 2,
              "evidence_onset_examples": (steps // 3) // 2,
              "optimizer": {"name": "AdamW", "learning_rate": 5e-5,
                            "weight_decay": .01, "gradient_clip": 1.0},
              "trainable_lora_parameters": trainable,
              "fit_group_outcome_pairs": sum(map(len, group_pairs.values())),
              "fit_evidence_pairs": sum(map(len, evidence_pairs.values())),
              "fit_onset_pairs": sum(map(len, onset_pairs.values())),
              "fit_scenes": len(group_scenes),
              "source_sha256": {
                  "group_manifest": group_sha,
                  "group_collection_audit": digest(args.group_collection_audit),
                  "group_state_audit": digest(args.group_state_audit),
                  "expert_manifest": expert_sha,
                  "evidence_manifest": evidence_sha,
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
        checkpoint = {"schema": "group4_evidence_onset_lora_v1",
                      "adapter": adapter, "source_sha256": report["source_sha256"],
                      "training_microsteps": steps,
                      "model_config_sha256": report["source_sha256"]["model_config"]}
        temporary = args.output / "final_adapter.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output / "final_adapter.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
