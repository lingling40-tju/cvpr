"""Small LoRA pilot for instruction-grounded visual progress.

Uses train-only goal panoramas and group-four policy success/failure pairs.
Fit/development scenes follow the frozen temporal-v2 split. The seed-33
episode-disjoint set and the v2 audit scenes are excluded from training and
model selection. No navigation policy update occurs here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)
from transformers import AutoModel, AutoProcessor


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def encode_images(model, processor, paths: list[Path], batch_size: int = 64) -> torch.Tensor:
    parts = []
    for start in range(0, len(paths), batch_size):
        opened = []
        try:
            for path in paths[start:start + batch_size]:
                opened.append(Image.open(path).convert("RGB"))
            batch = processor(images=opened, return_tensors="pt")
            pixels = batch["pixel_values"].to("cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vector = model.get_image_features(pixel_values=pixels)
            parts.append(F.normalize(vector.float(), dim=-1))
        finally:
            for image in opened:
                image.close()
    return torch.cat(parts)


def encode_texts(model, processor, texts: list[str], batch_size: int = 64) -> torch.Tensor:
    parts = []
    for start in range(0, len(texts), batch_size):
        # SigLIP pools the final sequence position. Its pad token is EOS, so
        # dynamic padding makes the embedding depend on other batch members.
        batch = processor(text=texts[start:start + batch_size],
                          padding="max_length", max_length=64,
                          truncation=True, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vector = model.get_text_features(**{key: value.to("cuda:0")
                                                for key, value in batch.items()})
        parts.append(F.normalize(vector.float(), dim=-1))
    return torch.cat(parts)


def score(image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    return (image * text).sum(-1)


def load_data(args) -> dict:
    policy = json.loads(args.policy_manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    goal = json.loads(args.goal_manifest.read_text())
    novel = json.loads(args.novel_manifest.read_text())
    if v2["source_manifest_sha256"] != digest(args.policy_manifest) or \
            v2["group_size"] != 4 or \
            not novel["selection"]["episode_disjoint_from_prior"]:
        raise ValueError("policy/novel provenance invalid")
    heldout_episode_ids = {str(row["episode_id"]) for row in novel["pairs"]}
    split_scenes = {name: set(v2["scene_split"][name])
                    for name in ("fit", "development", "audit")}
    if any(split_scenes[a] & split_scenes[b]
           for a, b in (("fit", "development"), ("fit", "audit"),
                        ("development", "audit"))):
        raise ValueError("scene leakage")
    goal_records = {}
    for section in ("fit", "calibration"):
        root = args.goal_root / section
        for eid in goal["subsets"][section]["episode_ids"]:
            path = root / "records" / f"{eid}.json"
            record = json.loads(path.read_text())
            if str(record["episode_id"]) != str(eid) or len(record["views"]) != 4:
                raise ValueError(f"goal record mismatch {eid}")
            goal_records[str(eid)] = {"scene": record["scene_id"],
                                      "instruction": record["instruction"],
                                      "images": [root / view["image"] for view in record["views"]]}
    goal_by_scene = defaultdict(list)
    dev_goal_by_scene = defaultdict(list)
    for section in ("fit", "calibration"):
        for pair in goal["subsets"][section]["pairs"]:
            a, b = str(pair["left"]), str(pair["right"])
            if a not in goal_records or b not in goal_records or \
                    goal_records[a]["scene"] != pair["scene"] or \
                    goal_records[b]["scene"] != pair["scene"]:
                raise ValueError("goal pair identity mismatch")
            if pair["scene"] in split_scenes["fit"] and \
                    a not in heldout_episode_ids and b not in heldout_episode_ids:
                goal_by_scene[pair["scene"]].append((a, b))
            elif pair["scene"] in split_scenes["development"]:
                dev_goal_by_scene[pair["scene"]].append((a, b))
    if len(goal_by_scene) < 30 or len(dev_goal_by_scene) != 8:
        raise ValueError("insufficient scene-balanced goal pairs")
    # Four fixed natural pairs per development scene bound repeated evaluation.
    dev_goal = [(scene, a, b)
                for scene in sorted(dev_goal_by_scene)
                for a, b in sorted(dev_goal_by_scene[scene],
                                   key=lambda p: rank("goal-dev-v1:" + str(p)))[:4]]
    if len(dev_goal) != 32:
        raise ValueError("incomplete development goal subset")
    v2_by_id = {row["pair_id"]: row for row in v2["pairs"]}
    policy_by_scene = defaultdict(list)
    dev_policy = []
    records = {}
    for pair in policy["pairs"]:
        assignment = v2_by_id[pair["pair_id"]]
        if pair["scene_id"] != assignment["scene_id"]:
            raise ValueError("policy/v2 scene mismatch")
        for role in ("success", "failure"):
            record_id = pair["pair_id"] + "_" + role
            record = json.loads((args.policy_frames_root / "records" /
                                 f"{record_id}.json").read_text())
            if record["record_id"] != record_id or len(record["frames"]) != 4:
                raise ValueError(f"policy image record mismatch {record_id}")
            records[record_id] = [args.policy_frames_root / frame["image"]
                                  for frame in record["frames"]]
        if assignment["split"] == "fit":
            policy_by_scene[pair["scene_id"]].append(pair)
        elif assignment["split"] == "development":
            dev_policy.append(pair)
    if len(dev_policy) != v2["counts"]["development"] or \
            sum(map(len, policy_by_scene.values())) != v2["counts"]["fit"]:
        raise ValueError("policy split count mismatch")
    return {"goal_records": goal_records,
            "goal_by_scene": goal_by_scene,
            "dev_goal": dev_goal,
            "policy_by_scene": policy_by_scene,
            "dev_policy": dev_policy,
            "v2_by_id": v2_by_id,
            "policy_images": records,
            "goal_manifest_sha256": digest(args.goal_manifest),
            "policy_manifest_sha256": digest(args.policy_manifest),
            "v2_manifest_sha256": digest(args.v2_manifest),
            "novel_manifest_sha256": digest(args.novel_manifest)}


def training_step(model, processor, data: dict, rng: random.Random,
                  batch_pairs: int = 8) -> dict:
    goal_scenes = sorted(data["goal_by_scene"])
    policy_scenes = sorted(data["policy_by_scene"])
    goal_pairs = [rng.choice(data["goal_by_scene"][rng.choice(goal_scenes)])
                  for _ in range(batch_pairs)]
    policy_pairs = [rng.choice(data["policy_by_scene"][rng.choice(policy_scenes)])
                    for _ in range(batch_pairs)]
    goal_images, goal_texts = [], []
    for a, b in goal_pairs:
        for eid in (a, b):
            record = data["goal_records"][eid]
            goal_images.append(rng.choice(record["images"]))
            goal_texts.append(record["instruction"])
    policy_images, policy_texts, wrong_texts = [], [], []
    for pair in policy_pairs:
        pid = pair["pair_id"]
        success = data["policy_images"][pid + "_success"]
        failure = data["policy_images"][pid + "_failure"]
        policy_images.extend((rng.choice(success[-2:]), rng.choice(failure[-2:]),
                              success[0]))
        policy_texts.append(pair["instruction"])
        wrong_texts.append(data["v2_by_id"][pid]["swapped_instruction"])
    images = encode_images(model, processor, goal_images + policy_images)
    texts = encode_texts(model, processor,
                         goal_texts + policy_texts + wrong_texts)
    n = batch_pairs
    gi = images[:2 * n].reshape(n, 2, -1)
    gt = texts[:2 * n].reshape(n, 2, -1)
    ps = images[2 * n:].reshape(n, 3, -1)
    pt = texts[2 * n:3 * n]
    wt = texts[3 * n:4 * n]
    goal_diffs = torch.cat((score(gi[:, 0], gt[:, 0]) - score(gi[:, 0], gt[:, 1]),
                            score(gi[:, 1], gt[:, 1]) - score(gi[:, 1], gt[:, 0]),
                            score(gi[:, 0], gt[:, 0]) - score(gi[:, 1], gt[:, 0]),
                            score(gi[:, 1], gt[:, 1]) - score(gi[:, 0], gt[:, 1])))
    policy_diff = score(ps[:, 0], pt) - score(ps[:, 1], pt)
    instruction_diff = score(ps[:, 0], pt) - score(ps[:, 0], wt)
    progress_diff = score(ps[:, 0], pt) - score(ps[:, 2], pt)
    goal_loss = F.softplus((.08 - goal_diffs) / .05).mean()
    policy_loss = F.softplus((.08 - policy_diff) / .05).mean()
    instruction_loss = F.softplus((.08 - instruction_diff) / .05).mean()
    progress_loss = F.softplus((.04 - progress_diff) / .05).mean()
    loss = goal_loss + policy_loss + instruction_loss + .2 * progress_loss
    return {"loss": loss, "goal": goal_loss, "policy": policy_loss,
            "instruction": instruction_loss, "progress": progress_loss}


@torch.no_grad()
def development(model, processor, data: dict) -> dict:
    model.eval()
    goal_images, goal_texts = [], []
    for _, a, b in data["dev_goal"]:
        for eid in (a, b):
            record = data["goal_records"][eid]
            goal_images.append(record["images"][0])
            goal_texts.append(record["instruction"])
    gi = encode_images(model, processor, goal_images)
    gt = encode_texts(model, processor, goal_texts)
    gi, gt = gi.reshape(-1, 2, 768), gt.reshape(-1, 2, 768)
    goal_hits = torch.cat((score(gi[:, 0], gt[:, 0]) > score(gi[:, 0], gt[:, 1]),
                           score(gi[:, 1], gt[:, 1]) > score(gi[:, 1], gt[:, 0])))
    policy_images, correct_texts, wrong_texts = [], [], []
    for pair in data["dev_policy"]:
        pid = pair["pair_id"]
        policy_images.extend((data["policy_images"][pid + "_success"][-1],
                              data["policy_images"][pid + "_failure"][-1]))
        correct_texts.append(pair["instruction"])
        wrong_texts.append(data["v2_by_id"][pid]["swapped_instruction"])
    pi = encode_images(model, processor, policy_images).reshape(-1, 2, 768)
    correct = encode_texts(model, processor, correct_texts)
    wrong = encode_texts(model, processor, wrong_texts)
    endpoint = score(pi[:, 0], correct) > score(pi[:, 1], correct)
    grounded = score(pi[:, 0], correct) > score(pi[:, 0], wrong)
    return {"goal_pairs": len(data["dev_goal"]),
            "goal_matching": float(goal_hits.float().mean()),
            "policy_pairs": len(data["dev_policy"]),
            "policy_success_over_failure": float(endpoint.float().mean()),
            "policy_correct_instruction": float(grounded.float().mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--novel-manifest", type=Path, required=True)
    parser.add_argument("--goal-manifest", type=Path, required=True)
    parser.add_argument("--goal-root", type=Path, required=True)
    parser.add_argument("--policy-frames-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2048 or not 1 <= args.eval_every <= args.steps:
        raise ValueError("invalid step/evaluation budget")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    data = load_data(args)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    base = AutoModel.from_pretrained(str(args.model), local_files_only=True,
                                     torch_dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"])).cuda().train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 589824:
        raise ValueError(f"unexpected LoRA trainable parameter count {trainable}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, weight_decay=.01)
    best = None
    baseline = development(model, processor, data)
    print(json.dumps({"step": 0, **baseline}), flush=True)
    history = [{"step": 0, **baseline}]
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        losses = training_step(model, processor, data, rng)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite LoRA loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == 1:
            print(json.dumps({"step": step,
                              "loss": {k: float(v.detach().cpu()) for k, v in losses.items()},
                              "elapsed_s": time.time() - started}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            dev = development(model, processor, data)
            # Balance the three requirements rather than maximizing one.
            key = (min(dev["goal_matching"] - .70,
                       dev["policy_success_over_failure"] - .70,
                       dev["policy_correct_instruction"] - .70),
                   dev["goal_matching"] + dev["policy_success_over_failure"] +
                   dev["policy_correct_instruction"], -step)
            history.append({"step": step, **dev})
            print(json.dumps(history[-1]), flush=True)
            if best is None or key > best[0]:
                best = (key, step, copy.deepcopy({
                    k: v.detach().cpu() for k, v in
                    get_peft_model_state_dict(model).items()}), dev)
    assert best is not None
    set_peft_model_state_dict(model, best[2])
    gate = {"goal_matching_at_least_0_70": best[3]["goal_matching"] >= .70,
            "policy_ranking_at_least_0_70":
                best[3]["policy_success_over_failure"] >= .70,
            "instruction_grounding_at_least_0_70":
                best[3]["policy_correct_instruction"] >= .70}
    report = {"schema": "siglip_lora_goal_policy_development_v2_fixed64",
              "interpretation": "Train-scene development screen; audit and seed33 novel pairs untouched; no RL.",
              "seed": args.seed, "steps": args.steps,
              "text_padding": "fixed_max_length_64",
              "selected_step": best[1], "trainable_parameters": trainable,
              "model_config_sha256": digest(args.model / "config.json"),
              "policy_manifest_sha256": data["policy_manifest_sha256"],
              "v2_manifest_sha256": data["v2_manifest_sha256"],
              "novel_manifest_sha256": data["novel_manifest_sha256"],
              "goal_manifest_sha256": data["goal_manifest_sha256"],
              "development": best[3], "predeclared_gate": gate,
              "baseline_development": baseline,
              "history": history, "elapsed_s": time.time() - started}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development.json").write_text(
        json.dumps(report, indent=2) + "\n")
    torch.save({"adapter": best[2], "selected_step": best[1],
                "text_padding": "fixed_max_length_64",
                "model_config_sha256": report["model_config_sha256"],
                "policy_manifest_sha256": report["policy_manifest_sha256"],
                "v2_manifest_sha256": report["v2_manifest_sha256"]},
               args.output_dir / "adapter.pt")
    print(json.dumps({"selected_step": best[1], "development": best[3],
                      "predeclared_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
