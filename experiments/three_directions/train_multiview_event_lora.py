"""Joint two-RGB instruction-conditioned arrival transition screen.

Only two rendered observations and natural instruction text reach the model.
Simulator distances and pair categories are labels only. The reserved audit
scenes are never loaded. This is an offline representation screen.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import copy
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

from history_grounding_lora import MAX_PIXELS, digest, load_model


STEPS = (256, 512, 768, 1024)
ACCUMULATION = 4
CATEGORIES = ("crossing", "far_nonarrival", "retreat",
              "wrong_instruction")
NEGATIVES = CATEGORIES[1:]
MARGIN = .5


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def load_part(part: str, capture: dict, labels: dict, rgb: Path,
              manifest_sha: str) -> dict[str, list[dict]]:
    if part not in ("fit", "development"):
        raise ValueError("reserved train-audit scenes cannot be opened")
    records = {}
    for plan in capture["selected"][part]:
        rid = plan["record_id"]
        record = read(rgb / part / "records" / f"{rid}.json")
        if rid in records or plan["scene_id"] not in capture[
                "scene_split"][part] or \
                record["schema"] != "multiview_event_rgb_input_v1" or \
                record["capture_manifest_sha256"] != manifest_sha or \
                record["record_id"] != rid or \
                set(record["input"]) != {"instruction", "terminal_clause",
                                         "wrong_instruction",
                                         "wrong_terminal_clause",
                                         "images_by_state"} or \
                any(record["input"][field] != plan[field] for field in (
                    "instruction", "terminal_clause", "wrong_instruction",
                    "wrong_terminal_clause")) or \
                set(record["input"]["images_by_state"]) != \
                    {str(i) for i in plan["states_to_capture"]}:
            raise ValueError(f"changed exact event RGB input: {rid}")
        records[rid] = {"plan": plan, "record": record,
                        "root": rgb / part}
    result = {kind: [] for kind in CATEGORIES}
    seen = set()
    for label in labels["selected"][part]:
        rid, kind = label["record_id"], label["kind"]
        if kind not in result or rid not in records or \
                label["pair_id"] != rid + ":" + kind or \
                label["pair_id"] in seen or \
                any(str(label[f"{position}_state_index"]) not in records[
                    rid]["record"]["input"]["images_by_state"] for
                    position in ("before", "after")):
            raise ValueError(f"changed event pair: {rid}/{kind}")
        seen.add(label["pair_id"])
        sample = {**records[rid], "label": label}
        result[kind].append(sample)
    if sum(len(v) for v in result.values()) != len(labels["selected"][part]):
        raise ValueError("incomplete event pair labels")
    return result


def inputs_for(processor, item: dict):
    from verl.utils.dataset.vision_utils import process_image

    label = item["label"]
    inp = item["record"]["input"]
    wrong = label["kind"] == "wrong_instruction"
    instruction = inp["wrong_instruction"] if wrong else inp["instruction"]
    clause = inp["wrong_terminal_clause"] if wrong else inp[
        "terminal_clause"]
    if not instruction or not clause or "<image>" in instruction or \
            "<image>" in clause:
        raise ValueError("invalid natural event instruction")
    text = ("Follow the route instruction and compare these two "
            "first-person views from the same route.\n"
            f"Instruction: {instruction}\n"
            f"Destination clause: {clause}\n"
            "Earlier view:\n<image>\nLater view:\n<image>\n"
            "Does the later view show stronger evidence that the "
            "destination has been reached?\n")
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True)
    prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", prompt,
                    flags=re.S)
    prompt = prompt.replace(
        "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    frames = []
    for position in ("before", "after"):
        index = str(label[f"{position}_state_index"])
        path = item["root"] / inp["images_by_state"][index]
        with Image.open(path) as source:
            raw = source.convert("RGB")
        try:
            buffer = io.BytesIO()
            raw.save(buffer, format="PNG")
            frames.append(process_image({"bytes": buffer.getvalue(),
                                         "max_pixels": MAX_PIXELS,
                                         "min_pixels": 1024}))
        finally:
            raw.close()
    try:
        value = processor(text=[prompt], images=frames, return_tensors="pt")
    finally:
        for frame in frames:
            if hasattr(frame, "close"):
                frame.close()
    if int(value["input_ids"].shape[1]) > 4096:
        raise ValueError("two-view context exceeds 4096 tokens")
    return value.to("cuda")


def score(processor, model, head, item: dict) -> torch.Tensor:
    inputs = inputs_for(processor, item)
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    value, _ = head(output.logits[0, -1])
    return value


def index_by_scene_episode(items: list[dict]) -> dict:
    result = defaultdict(lambda: defaultdict(list))
    for item in items:
        plan = item["plan"]
        result[plan["scene_id"]][str(plan["episode_id"])].append(item)
    return result


def sample_index(index: dict, rng: random.Random) -> dict:
    scene = rng.choice(list(index))
    episode = rng.choice(list(index[scene]))
    return rng.choice(index[scene][episode])


def loss_for(processor, model, head, positive: dict,
             negative: dict) -> torch.Tensor:
    if positive["label"]["kind"] != "crossing" or \
            negative["label"]["kind"] not in NEGATIVES:
        raise ValueError("invalid binary event pair")
    pos = score(processor, model, head, positive)
    neg = score(processor, model, head, negative)
    return (F.softplus(MARGIN - (pos-neg)) + .25*F.softplus(-pos) +
            .25*F.softplus(neg))


def fixed_subset(part: dict[str, list[dict]]) -> list[dict]:
    selected = []
    for kind in CATEGORIES:
        by_scene = defaultdict(list)
        for item in part[kind]:
            by_scene[item["plan"]["scene_id"]].append(item)
        for scene, rows in sorted(by_scene.items()):
            rows.sort(key=lambda x: hashlib.sha256(
                ("two-view-checkpoint-v1:" + x["label"]["pair_id"]).encode()
            ).hexdigest())
            selected.extend(rows[:8])
    if len(selected) < 180:
        raise ValueError("small two-view checkpoint subset")
    return selected


def auc(positive: list[float], negative: list[float]) -> float:
    return sum((p > n) + .5*(p == n) for p in positive for n in negative)/(
        len(positive)*len(negative))


def threshold_for(positive: list[float], negative: list[float]) -> dict:
    candidates = sorted(set(positive + negative), reverse=True)
    feasible = []
    for value in [max(candidates)+1.] + candidates:
        fpr = sum(x >= value for x in negative)/len(negative)
        if fpr <= .05:
            feasible.append((sum(x >= value for x in positive)/
                             len(positive), -fpr, value))
    recall, negative_fpr, threshold = max(feasible)
    return {"threshold": threshold, "pooled_recall": recall,
            "pooled_fpr": -negative_fpr, "target_fpr": .05}


def macro(rows: list[dict], group: str, threshold: float) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[group])].append(row["score"] >= threshold)
    return {"groups": len(grouped),
            "macro_rate": sum(sum(x)/len(x) for x in grouped.values())/
                          len(grouped)}


def evaluate(processor, model, head, items: list[dict], full: bool) -> tuple[dict, list[dict]]:
    model.eval()
    head.eval()
    scored = []
    with torch.inference_mode():
        for item in items:
            label, plan = item["label"], item["plan"]
            value = float(score(processor, model, head, item))
            if not math.isfinite(value):
                raise ValueError(f"nonfinite event score {label['pair_id']}")
            scored.append({"pair_id": label["pair_id"],
                           "record_id": label["record_id"],
                           "episode_id": str(plan["episode_id"]),
                           "scene_id": plan["scene_id"],
                           "class": label["kind"], "score": value,
                           "task_success_for_audit_only": label[
                               "task_success_for_audit_only"]})
    by_class = {kind: [x for x in scored if x["class"] == kind]
                for kind in CATEGORIES}
    if any(not x for x in by_class.values()):
        raise ValueError("missing two-view development class")
    positives = by_class["crossing"]
    negatives = [x for kind in NEGATIVES for x in by_class[kind]]
    threshold = threshold_for([x["score"] for x in positives],
                              [x["score"] for x in negatives])
    value = threshold["threshold"]
    near_failure = [x for x in positives if not x[
        "task_success_for_audit_only"]]
    if not near_failure:
        raise ValueError("missing near-failure development positives")
    rates = {
        "near_failure_recall": sum(x["score"] >= value for x in
                                   near_failure)/len(near_failure),
        "far_nonarrival_fpr": sum(x["score"] >= value for x in
                                  by_class["far_nonarrival"])/len(
                                      by_class["far_nonarrival"]),
        "retreat_fpr": sum(x["score"] >= value for x in
                           by_class["retreat"])/len(by_class["retreat"]),
        "wrong_instruction_fpr": sum(x["score"] >= value for x in
                                     by_class["wrong_instruction"])/len(
                                         by_class["wrong_instruction"]),
    }
    by_record = defaultdict(dict)
    for row in scored:
        by_record[row["record_id"]][row["class"]] = row["score"]
    same_route = [(values["crossing"], values[kind])
                  for values in by_record.values()
                  if "crossing" in values for kind in NEGATIVES
                  if kind in values]
    if not same_route:
        raise ValueError("missing within-route event contrasts")
    within_route_rank = sum(p > n for p, n in same_route)/len(same_route)
    source_ids = {kind: len({x["episode_id"] for x in by_class[kind]})
                  for kind in CATEGORIES}
    gates = {
        "crossing_episode_ids_at_least_50": source_ids["crossing"] >= 50,
        "far_episode_ids_at_least_50": source_ids["far_nonarrival"] >= 50,
        "retreat_episode_ids_at_least_25": source_ids["retreat"] >= 25,
        "wrong_episode_ids_at_least_30": source_ids[
            "wrong_instruction"] >= 30,
        "pooled_fpr_at_most_0_05": threshold["pooled_fpr"] <= .05,
        "crossing_recall_at_least_0_55": threshold[
            "pooled_recall"] >= .55,
        "near_failure_recall_at_least_0_50": rates[
            "near_failure_recall"] >= .50,
        "retreat_fpr_at_most_0_10": rates["retreat_fpr"] <= .10,
        "wrong_fpr_at_most_0_12": rates[
            "wrong_instruction_fpr"] <= .12,
    }
    result = {"pairs": len(scored),
              "counts": {kind: len(by_class[kind]) for kind in CATEGORIES},
              "unique_episode_ids": source_ids,
              "auc_pooled": auc([x["score"] for x in positives],
                                [x["score"] for x in negatives]),
              "within_route_rank_accuracy": within_route_rank,
              "threshold": threshold, "rates": rates,
              "model_gates": gates if full else None}
    if full:
        result["episode_macro"] = {
            kind: macro(by_class[kind], "episode_id", value)
            for kind in CATEGORIES}
        result["scene_macro"] = {
            kind: macro(by_class[kind], "scene_id", value)
            for kind in CATEGORIES}
    return result, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("capture-manifest", "pair-labels", "source-report",
                 "verification", "rgb-root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.set_num_threads(6)
    rng = random.Random(11)
    capture, labels, source, verified = map(
        read, (args.capture_manifest, args.pair_labels,
               args.source_report, args.verification))
    capture_sha = digest(args.capture_manifest)
    if capture["schema"] != "multiview_event_rgb_capture_manifest_v1" or \
            labels["schema"] != "multiview_event_privileged_pair_labels_v1" or \
            labels["capture_manifest_sha256"] != capture_sha or \
            source["capture_manifest_sha256"] != capture_sha or \
            verified["capture_manifest_sha256"] != capture_sha or \
            verified["smoke_record_id"] is not None or \
            [p["records"] for p in verified["parts"]] != [1391, 351] or \
            verified["audit_opened"] or not source["ready_for_rgb_replay"]:
        raise ValueError("changed or unverified event source")
    fit = load_part("fit", capture, labels, args.rgb_root, capture_sha)
    dev = None if args.smoke else load_part(
        "development", capture, labels, args.rgb_root, capture_sha)
    for part, sample in (("fit", fit), ("development", dev)):
        if sample is None:
            continue
        if {k: len(v) for k, v in sample.items()} != source["counts"][part]:
            raise ValueError(f"changed {part} pair counts")
    fit_index = {kind: index_by_scene_episode(fit[kind]) for kind in
                 CATEGORIES}
    pos_by_rid = {x["label"]["record_id"]: x for x in fit["crossing"]}
    if len(pos_by_rid) != len(fit["crossing"]):
        raise ValueError("duplicate positive crossing")
    subset = None if args.smoke else fixed_subset(dev)
    processor, model, head, trainable = load_model(args.model)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": 1e-4},
        {"params": list(head.parameters()), "lr": 3e-4}],
        weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    steps = 4 if args.smoke else STEPS[-1]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    best = None
    history = []
    for step in range(1, steps+1):
        model.train()
        head.train()
        kind = NEGATIVES[(step-1) % len(NEGATIVES)]
        negative = sample_index(fit_index[kind], rng)
        positive = pos_by_rid.get(negative["label"]["record_id"])
        if positive is None:
            scene = negative["plan"]["scene_id"]
            positive = sample_index({scene: fit_index["crossing"][scene]},
                                    rng)
        loss = loss_for(processor, model, head, positive, negative)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite event loss at {step}")
        (loss/ACCUMULATION).backward()
        if step % ACCUMULATION == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(head.parameters()), 1.0)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise ValueError(f"invalid event gradient at {step}: {norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "elapsed_seconds": time.time()-started}),
                  flush=True)
        if not args.smoke and step in STEPS:
            metric, _ = evaluate(processor, model, head, subset, full=False)
            history.append({"step": step, **metric})
            quality = (metric["auc_pooled"],
                       metric["within_route_rank_accuracy"], -step)
            print(json.dumps({"checkpoint": step,
                              "subset_auc": metric["auc_pooled"],
                              "subset_recall": metric["threshold"][
                                  "pooled_recall"]}), flush=True)
            if best is None or quality > best[0]:
                state = {
                    "capture_manifest_sha256": capture_sha,
                    "selected_step": step, "source_model": str(args.model),
                    "adapter": {k: v.detach().cpu().clone() for k, v in
                                get_peft_model_state_dict(model).items()},
                    "head": {k: v.detach().cpu().clone() for k, v in
                             head.state_dict().items()},
                }
                temp = args.output / "selected.tmp"
                torch.save(state, temp)
                os.replace(temp, args.output / "selected.pt")
                best = (quality, step)
    if args.smoke:
        write(args.output / "smoke.json", {
            "schema": "multiview_event_lora_smoke_v1",
            "capture_manifest_sha256": capture_sha,
            "steps": steps, "optimizer_updates": steps//ACCUMULATION,
            "lora_parameters": trainable,
            "fit_pairs": sum(len(v) for v in fit.values()),
            "development_loaded": False, "audit_loaded": False,
            "elapsed_seconds": time.time()-started})
        return
    state = torch.load(args.output / "selected.pt", map_location="cpu",
                       weights_only=False)
    if best is None or state["capture_manifest_sha256"] != capture_sha or \
            state["selected_step"] != best[1]:
        raise ValueError("selected event checkpoint changed")
    set_peft_model_state_dict(model, state["adapter"])
    head.load_state_dict(state["head"])
    full_dev = [x for kind in CATEGORIES for x in dev[kind]]
    metrics, scores = evaluate(processor, model, head, full_dev, full=True)
    report = {
        "schema": "multiview_event_lora_development_v1",
        "capture_manifest_sha256": capture_sha,
        "pair_labels_sha256": digest(args.pair_labels),
        "rgb_verification_sha256": digest(args.verification),
        "model": str(args.model),
        "selected_step": best[1], "planned_steps": list(STEPS),
        "lora_parameters": trainable,
        "fit_pairs": sum(len(v) for v in fit.values()),
        "development": metrics,
        "model_gate_passed_before_controls": all(metrics[
            "model_gates"].values()),
        "checkpoint_history": history,
        "elapsed_seconds": time.time()-started,
        "audit_opened": False, "navigation_result": False,
    }
    write(args.output / "development_scores.json", {
        "schema": "multiview_event_development_scores_v1",
        "capture_manifest_sha256": capture_sha,
        "selected_step": best[1], "scores": scores})
    write(args.output / "development.json", report)
    print(json.dumps({"selected_step": best[1],
                      "development": metrics,
                      "model_gate_passed_before_controls": report[
                          "model_gate_passed_before_controls"],
                      "elapsed_seconds": report["elapsed_seconds"]}),
          flush=True)


if __name__ == "__main__":
    main()
