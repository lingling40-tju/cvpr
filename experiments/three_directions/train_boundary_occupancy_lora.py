"""Fit-only instruction-conditioned goal-region occupancy screen.

The scorer sees one current RGB view, instruction, and recent executed
motion. Outside/inside roles and simulator distances never enter prompts.
Development scenes select a checkpoint and threshold. Audit is unopened.
This offline screen is not a navigation reward or result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time

from PIL import Image
import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from history_grounding_lora import FORMAT, MAX_PIXELS, digest, load_model


STEPS = (256, 512, 768)
ACCUMULATION = 4
MARGIN = .5


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_part(part: str, manifest: dict, labels: dict, replay: Path,
              manifest_sha: str) -> list[dict]:
    if part not in ("fit", "development"):
        raise ValueError("audit scenes cannot be loaded for model selection")
    labels_by_id = {x["record_id"]: x for x in labels["selected"][part]}
    scenes = set(manifest["scene_split"][part])
    result = []
    for plan in manifest["selected"][part]:
        rid = plan["record_id"]
        record = read(replay / part / "records" / f"{rid}.json")
        if plan["scene_id"] not in scenes or \
                record["schema"] != "boundary_occupancy_rgb_model_input_v1" or \
                record["manifest_sha256"] != manifest_sha or \
                record["record_id"] != rid or \
                set(record["input"]) != {"instruction", "wrong_instruction",
                                         "images", "action_history_by_state"} or \
                record["input"]["instruction"] != plan["instruction"] or \
                record["input"]["wrong_instruction"] != plan["wrong_instruction"]:
            raise ValueError(f"changed replay input: {rid}")
        if rid not in labels_by_id or \
                not 3.5 <= labels_by_id[rid]["outside_distance_m"] <= 4.5 or \
                not 0 <= labels_by_id[rid]["inside_distance_m"] <= 3.0:
            raise ValueError(f"changed boundary label: {rid}")
        result.append({"plan": plan, "record": record,
                       "label": labels_by_id[rid], "root": replay / part})
    if len(result) != len(labels_by_id):
        raise ValueError(f"incomplete selected replay: {part}")
    return result


def prompt_text(instruction: str, history: list[dict]) -> str:
    """No state role, distance, outcome, or absolute turn number is present."""
    recent = history[-4:]
    actions = " | ".join(", ".join(turn["executed_actions"])
                         for turn in recent) or "none"
    if not instruction or "<image>" in instruction or \
            any("stop" in turn["executed_actions"] for turn in recent):
        raise ValueError("invalid observation-only model input")
    return ("After that, the observation is:\n<image>\n"
            f"Instruction: {instruction}\n"
            f"Recent executed motion actions, earliest to latest: {actions}\n"
            "Decide your next action.\n" + FORMAT)


def model_inputs(processor, item: dict, state: str,
                 instruction: str | None = None):
    from verl.utils.dataset.vision_utils import process_image

    inp = item["record"]["input"]
    instruction = instruction or inp["instruction"]
    history = inp["action_history_by_state"][state]
    raw_text = prompt_text(instruction, history)
    messages = [{"role": "user", "content": raw_text}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", prompt,
                    flags=re.S)
    prompt = prompt.replace(
        "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    path = item["root"] / inp["images"][state]
    with Image.open(path) as source:
        frame = source.convert("RGB")
    try:
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        processed = process_image({"bytes": buffer.getvalue(),
                                   "max_pixels": MAX_PIXELS,
                                   "min_pixels": 1024})
        try:
            inputs = processor(text=[prompt], images=[processed],
                               return_tensors="pt")
        finally:
            if hasattr(processed, "close"):
                processed.close()
    finally:
        frame.close()
    if int(inputs["input_ids"].shape[1]) > 4096:
        raise ValueError("occupancy context exceeds 4096 tokens")
    return inputs.to("cuda")


def score(processor, model, head, item: dict, state: str,
          instruction: str | None = None) -> torch.Tensor:
    inputs = model_inputs(processor, item, state, instruction)
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    value, _ = head(output.logits[0, -1])
    return value


def sample_scene_balanced(index: dict, rng: random.Random) -> dict:
    scene = rng.choice(list(index))
    episode = rng.choice(list(index[scene]))
    return rng.choice(index[scene][episode])


def index_by_scene_episode(items: list[dict]) -> dict:
    result = defaultdict(lambda: defaultdict(list))
    for item in items:
        plan = item["plan"]
        result[plan["scene_id"]][str(plan["episode_id"])].append(item)
    return result


def loss_for(processor, model, head, item: dict) -> torch.Tensor:
    correct = item["record"]["input"]["instruction"]
    inside = score(processor, model, head, item, "inside", correct)
    outside = score(processor, model, head, item, "outside", correct)
    loss = (F.softplus(MARGIN - (inside - outside)) +
            .25 * F.softplus(-inside) + .25 * F.softplus(outside))
    if item["plan"]["wrong_instruction_same_start"]:
        wrong = item["record"]["input"]["wrong_instruction"]
        if not wrong:
            raise ValueError("missing same-start instruction contrast")
        negative = score(processor, model, head, item, "inside", wrong)
        loss = (loss + F.softplus(MARGIN - (inside - negative)) +
                .25 * F.softplus(negative))
    return loss


def auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        raise ValueError("missing AUC class")
    return sum((p > n) + .5 * (p == n) for p in positive
               for n in negative) / (len(positive) * len(negative))


def fixed_dev_subset(items: list[dict]) -> list[dict]:
    by_scene = defaultdict(list)
    for item in items:
        by_scene[item["plan"]["scene_id"]].append(item)
    selected = []
    for scene, rows in sorted(by_scene.items()):
        rows.sort(key=lambda item: hashlib.sha256(
            ("boundary-occupancy-checkpoint-v1:" +
             item["plan"]["record_id"]).encode()).hexdigest())
        selected.extend(rows[:8])
    if len(selected) < 48 or len(by_scene) != 8:
        raise ValueError("insufficient scene-balanced development subset")
    return selected


def choose_threshold(positive: list[float], negative: list[float]) -> dict:
    candidates = sorted(set(positive + negative), reverse=True)
    feasible = []
    for threshold in [max(candidates) + 1.] + candidates:
        fpr = sum(x >= threshold for x in negative) / len(negative)
        if fpr <= .05:
            recall = sum(x >= threshold for x in positive) / len(positive)
            feasible.append((recall, -fpr, threshold))
    recall, negative_fpr, threshold = max(feasible)
    return {"threshold": threshold, "pooled_fpr": -negative_fpr,
            "pooled_recall": recall, "target_fpr": .05}


def group_rate(scores: list[dict], key: str, threshold: float,
               *, positive: bool) -> dict:
    by_group = defaultdict(list)
    for row in scores:
        by_group[str(row[key])].append(float(row["score"]) >= threshold)
    if not by_group:
        return {"groups": 0, "macro_rate": None}
    return {"groups": len(by_group),
            "macro_rate": sum(sum(values) / len(values) for values in
                              by_group.values()) / len(by_group),
            "interpretation": "recall" if positive else "false_positive_rate"}


def evaluate(processor, model, head, items: list[dict],
             full: bool) -> tuple[dict, list[dict]]:
    model.eval()
    head.eval()
    rows = []
    with torch.inference_mode():
        for item in items:
            plan = item["plan"]
            inp = item["record"]["input"]
            for cls, state, instruction in (
                    ("outside", "outside", inp["instruction"]),
                    ("inside", "inside", inp["instruction"]),
                    ("wrong", "inside", inp["wrong_instruction"])):
                if instruction is None:
                    continue
                value = float(score(processor, model, head, item,
                                    state, instruction))
                if not math.isfinite(value):
                    raise ValueError(f"nonfinite development score: {plan['record_id']}")
                rows.append({"record_id": plan["record_id"],
                             "episode_id": str(plan["episode_id"]),
                             "scene_id": plan["scene_id"], "class": cls,
                             "score": value,
                             "same_start_wrong": bool(
                                 plan["wrong_instruction_same_start"]),
                             "task_success_for_audit_only":
                                 item["label"]["task_success"]})
    by_id = defaultdict(dict)
    for row in rows:
        by_id[row["record_id"]][row["class"]] = row
    if len(by_id) != len(items) or \
            any(not {"outside", "inside"}.issubset(classes)
                for classes in by_id.values()):
        raise ValueError("incomplete development score coverage")
    positives = [x for x in rows if x["class"] == "inside"]
    negatives = [x for x in rows if x["class"] != "inside"]
    threshold = choose_threshold([x["score"] for x in positives],
                                 [x["score"] for x in negatives])
    value = threshold["threshold"]
    wrong = [x for x in rows if x["class"] == "wrong"]
    same_start_wrong = [x for x in wrong if x["same_start_wrong"]]
    near_failure = [x for x in positives if not x[
        "task_success_for_audit_only"]]
    rates = {
        "outside_fpr": sum(x["score"] >= value for x in rows
                           if x["class"] == "outside") / len(positives),
        "wrong_instruction_fpr": sum(x["score"] >= value for x in wrong) /
            len(wrong),
        "same_start_wrong_instruction_fpr": sum(
            x["score"] >= value for x in same_start_wrong) /
            len(same_start_wrong),
        "near_failure_recall": sum(x["score"] >= value for x in
                                   near_failure) / len(near_failure),
        "crossing_order_accuracy": sum(
            values["inside"]["score"] > values["outside"]["score"]
            for values in by_id.values()) / len(by_id),
        "instruction_order_accuracy": sum(
            values["inside"]["score"] > values["wrong"]["score"]
            for values in by_id.values() if "wrong" in values) / len(wrong),
    }
    gate = {
        "positive_episode_ids_at_least_50":
            len({x["episode_id"] for x in positives}) >= 50,
        "pooled_fpr_at_most_0_05": threshold["pooled_fpr"] <= .05,
        "pooled_recall_at_least_0_55": threshold["pooled_recall"] >= .55,
        "wrong_instruction_fpr_at_most_0_12":
            rates["wrong_instruction_fpr"] <= .12,
        "near_failure_recall_at_least_0_50":
            rates["near_failure_recall"] >= .50,
    }
    report = {"records": len(items),
              "positive_episode_ids": len({x["episode_id"] for x in positives}),
              "near_failure_episode_ids": len({x["episode_id"] for x in
                                                near_failure}),
              "same_start_wrong_episode_ids": len({x["episode_id"] for x in
                                                    same_start_wrong}),
              "counts": {kind: sum(x["class"] == kind for x in rows)
                         for kind in ("outside", "inside", "wrong")},
              "auc_pooled": auc([x["score"] for x in positives],
                                [x["score"] for x in negatives]),
              "threshold": threshold, "rates": rates,
              "gate": gate if full else None}
    if full:
        report["episode_macro"] = {
            "inside_recall": group_rate(positives, "episode_id", value,
                                        positive=True),
            "outside_fpr": group_rate([x for x in rows if x["class"] ==
                                        "outside"], "episode_id", value,
                                       positive=False),
            "wrong_fpr": group_rate(wrong, "episode_id", value,
                                    positive=False),
        }
        report["scene_macro"] = {
            "inside_recall": group_rate(positives, "scene_id", value,
                                        positive=True),
            "outside_fpr": group_rate([x for x in rows if x["class"] ==
                                        "outside"], "scene_id", value,
                                       positive=False),
            "wrong_fpr": group_rate(wrong, "scene_id", value,
                                    positive=False),
        }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "verification", "replay-root",
                 "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.set_num_threads(6)
    rng = random.Random(11)
    manifest, labels, verification = map(read, (args.manifest, args.labels,
                                                args.verification))
    manifest_sha = digest(args.manifest)
    if manifest["schema"] != "boundary_occupancy_rgb_replay_manifest_v1" or \
            labels["schema"] != "boundary_occupancy_privileged_labels_v1" or \
            labels["replay_manifest_sha256"] != manifest_sha or \
            verification["manifest_sha256"] != manifest_sha or \
            verification["smoke_record_id"] is not None or \
            [(p["part"], p["records"], p["max_geodesic_drift_m"])
             for p in verification["parts"]] != [
                ("fit", 1184, 0.0), ("development", 288, 0.0)]:
        raise ValueError("changed or unverified frozen source")
    fit = load_part("fit", manifest, labels, args.replay_root, manifest_sha)
    dev = None if args.smoke else load_part(
        "development", manifest, labels, args.replay_root, manifest_sha)
    index_all = index_by_scene_episode(fit)
    index_exact = index_by_scene_episode([x for x in fit if x["plan"][
        "wrong_instruction_same_start"]])
    if len(fit) != 1184 or len(index_all) != 38 or len(index_exact) < 20 or \
            (dev is not None and len(dev) != 288):
        raise ValueError("frozen fit/development scene coverage changed")
    processor, model, head, trainable = load_model(args.model)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": 1e-4},
        {"params": list(head.parameters()), "lr": 3e-4}],
        weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    steps = 4 if args.smoke else STEPS[-1]
    subset = None if args.smoke else fixed_dev_subset(dev)
    best, history = None, []
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        head.train()
        item = sample_scene_balanced(index_exact if step % 2 == 0 else
                                     index_all, rng)
        loss = loss_for(processor, model, head, item)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite occupancy loss at {step}")
        (loss / ACCUMULATION).backward()
        if step % ACCUMULATION == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(head.parameters()), 1.0)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise ValueError(f"invalid occupancy gradient at {step}: {norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}),
                  flush=True)
        if not args.smoke and step in STEPS:
            metrics, _ = evaluate(processor, model, head, subset, full=False)
            history.append({"step": step, **metrics})
            print(json.dumps({"checkpoint": step,
                              "subset_auc": metrics["auc_pooled"],
                              "subset_recall": metrics["threshold"][
                                  "pooled_recall"]}), flush=True)
            quality = (metrics["auc_pooled"],
                       metrics["rates"]["crossing_order_accuracy"], -step)
            if best is None or quality > best[0]:
                state = {"selected_step": step,
                         "manifest_sha256": manifest_sha,
                         "source_model": str(args.model),
                         "adapter": {k: v.detach().cpu().clone() for k, v in
                                     get_peft_model_state_dict(model).items()},
                         "head": {k: v.detach().cpu().clone() for k, v in
                                  head.state_dict().items()}}
                temporary = args.output / "selected.tmp"
                torch.save(state, temporary)
                os.replace(temporary, args.output / "selected.pt")
                best = (quality, step)
    if args.smoke:
        write(args.output / "smoke.json", {
            "schema": "boundary_occupancy_lora_smoke_v1",
            "manifest_sha256": manifest_sha, "steps": steps,
            "optimizer_updates": steps // ACCUMULATION,
            "lora_parameters": trainable,
            "fit_records": len(fit),
            "development_loaded": False, "audit_loaded": False,
            "elapsed_seconds": time.time() - started})
        return
    if best is None:
        raise ValueError("no occupancy checkpoint selected")
    state = torch.load(args.output / "selected.pt", map_location="cpu",
                       weights_only=False)
    if state["manifest_sha256"] != manifest_sha or \
            state["selected_step"] != best[1]:
        raise ValueError("checkpoint identity changed")
    set_peft_model_state_dict(model, state["adapter"])
    head.load_state_dict(state["head"])
    metrics, scores = evaluate(processor, model, head, dev, full=True)
    report = {"schema": "boundary_occupancy_lora_development_v1",
              "manifest_sha256": manifest_sha,
              "labels_sha256": digest(args.labels),
              "replay_verification_sha256": digest(args.verification),
              "model": str(args.model),
              "selected_step": best[1], "planned_steps": list(STEPS),
              "lora_parameters": trainable,
              "fit_records": len(fit), "fit_scenes": len(index_all),
              "development": metrics, "gate_passed": all(metrics["gate"].values()),
              "checkpoint_history": history,
              "elapsed_seconds": time.time() - started,
              "audit_opened": False, "navigation_result": False}
    write(args.output / "development_scores.json", {
        "schema": "boundary_occupancy_development_scores_v1",
        "manifest_sha256": manifest_sha,
        "selected_step": best[1], "scores": scores})
    write(args.output / "development.json", report)
    print(json.dumps({"selected_step": best[1],
                      "development": metrics, "gate_passed": report[
                          "gate_passed"], "elapsed_seconds": report[
                              "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
