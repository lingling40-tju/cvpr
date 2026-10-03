"""Train a small navigation-SFT LoRA on group-four outcome preferences.

The frozen SFT reference log probabilities are computed separately from
the exact same image and prompt. Only fit scenes update the adapter;
development scenes choose among predeclared checkpoints. No audit or
val-unseen example is accepted by this training CLI.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from history_grounding_lora import build_inputs, digest


LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
        "target_modules": ["q_proj", "v_proj"]}
BETA = .1
LR = 2e-5
GRAD_ACCUM = 4
EPOCHS = 2
CHECKPOINT_PAIRS = (256, 512, 948)


def load_reference(paths: list[Path], manifest: dict, part: str,
                   model_config_sha: str) -> dict[str, dict]:
    result = {}
    expected_image_sha = None
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["schema"] != "group4_first_action_sft_score_v1" or \
                payload["part"] != part or payload["smoke_limit"] or \
                payload["manifest_sha256"] != digest(manifest["path"]) or \
                payload["model_config_sha256"] != model_config_sha:
            raise ValueError(f"bad frozen SFT reference {path}")
        if expected_image_sha is None:
            expected_image_sha = payload["image_index_sha256"]
        if payload["image_index_sha256"] != expected_image_sha:
            raise ValueError("mixed image cache")
        for row in payload["rows"]:
            gid = row["group_id"]
            if gid in result:
                raise ValueError(f"duplicate reference group {gid}")
            result[gid] = row
    if set(result) != {row["group_id"] for row in manifest["data"]["selected"][part]}:
        raise ValueError(f"incomplete SFT reference {part}")
    return result


def response_logprob(model, processor, image_root: Path, image: dict,
                     response: str) -> torch.Tensor:
    item = {"root": image_root, "record": {
        "instruction": image["instruction"], "initial_image": image["image"],
        "turns": []}}
    prompt = build_inputs(processor, item, 0).to("cuda")
    prompt_len = int(prompt["input_ids"].shape[1])
    ids = processor.tokenizer.encode(response, add_special_tokens=False)
    if not ids or processor.tokenizer.eos_token_id is None:
        raise ValueError("empty response or missing EOS")
    suffix = torch.tensor([ids + [processor.tokenizer.eos_token_id]],
                          dtype=prompt["input_ids"].dtype, device="cuda")
    full = dict(prompt)
    full["input_ids"] = torch.cat((prompt["input_ids"], suffix), dim=1)
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    logits = model(**full, use_cache=False).logits[0,
        prompt_len - 1:prompt_len + suffix.shape[1] - 1].float()
    return -F.cross_entropy(logits, suffix[0], reduction="sum")


@torch.inference_mode()
def evaluate(model, processor, rows: list[dict], image_index: dict,
             image_root: Path, reference: dict[str, dict]) -> dict:
    model.eval()
    correct = 0
    ref_correct = 0
    margins = []
    outcomes = []
    for row in rows:
        eid = str(row["episode_id"])
        image = image_index["images"][eid]
        good = response_logprob(model, processor, image_root, image,
                                row["preferred_response"])
        bad = response_logprob(model, processor, image_root, image,
                               row["rejected_response"])
        margin = float((good - bad).cpu())
        ref = reference[row["group_id"]]
        ref_margin = (ref["preferred"]["with_eos_logprob_sum"] -
                      ref["rejected"]["with_eos_logprob_sum"])
        correct += margin > 0
        ref_correct += ref_margin > 0
        margins.append(margin - ref_margin)
        outcomes.append({"group_id": row["group_id"],
                         "episode_id": eid, "scene_id": row["scene_id"],
                         "margin": margin, "reference_margin": ref_margin})
    return {"pairs": len(rows), "correct": correct,
            "accuracy": correct / len(rows),
            "reference_correct": ref_correct,
            "reference_accuracy": ref_correct / len(rows),
            "mean_margin_improvement": sum(margins) / len(margins),
            "rows": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--fit-reference", type=Path, action="append", required=True)
    parser.add_argument("--dev-reference", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.set_num_threads(6)
    rng = random.Random(11)
    manifest_data = json.loads(args.manifest.read_text())
    if manifest_data["schema"] != "group4_first_action_preference_manifest_v1":
        raise ValueError("wrong group-four preference manifest")
    manifest = {"path": args.manifest, "data": manifest_data}
    image_index = {}
    for part in ("fit", "development"):
        path = args.image_root / part / "index.json"
        index = json.loads(path.read_text())
        if index["manifest_sha256"] != digest(args.manifest) or \
                index["groups"] != len(manifest_data["selected"][part]):
            raise ValueError(f"incomplete image index {part}")
        image_index[part] = index
    model_config_sha = digest(args.model / "config.json")
    fit_reference = load_reference(args.fit_reference, manifest, "fit", model_config_sha)
    dev_reference = load_reference(args.dev_reference, manifest, "development", model_config_sha)
    fit = manifest_data["selected"]["fit"]
    dev = manifest_data["selected"]["development"]
    counts = Counter(str(row["episode_id"]) for row in fit)
    sample_weight_mean = sum(1 / counts[str(row["episode_id"])] for row in fit) / len(fit)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16, local_files_only=True)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, LoraConfig(**LORA)).cuda()
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not 100_000 < sum(p.numel() for p in trainable) < 30_000_000:
        raise ValueError("unexpected LoRA parameter count")
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    selected = None
    seen = 0
    began = time.time()
    for epoch in range(EPOCHS):
        order = fit.copy()
        rng.shuffle(order)
        for row in order:
            seen += 1
            model.train()
            eid = str(row["episode_id"])
            image = image_index["fit"]["images"][eid]
            good = response_logprob(model, processor, args.image_root / "fit",
                                    image, row["preferred_response"])
            bad = response_logprob(model, processor, args.image_root / "fit",
                                   image, row["rejected_response"])
            reference = fit_reference[row["group_id"]]
            reference_margin = (reference["preferred"]["with_eos_logprob_sum"] -
                                reference["rejected"]["with_eos_logprob_sum"])
            weight = (1 / counts[eid]) / sample_weight_mean
            loss = weight * F.softplus(-BETA * ((good - bad) - reference_margin))
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"nonfinite DPO loss at pair {seen}")
            (loss / GRAD_ACCUM).backward()
            if seen % GRAD_ACCUM == 0:
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not bool(torch.isfinite(norm)) or float(norm) <= 0:
                    raise ValueError(f"invalid LoRA gradient at pair {seen}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if seen % 32 == 0 or seen == 1:
                print(json.dumps({"pairs_seen": seen, "epoch": epoch + 1,
                                  "loss": float(loss.detach().cpu()),
                                  "elapsed_seconds": time.time() - began}), flush=True)
            if args.smoke and seen == 4:
                print(json.dumps({"smoke_pairs": seen, "trainable_parameters":
                                  sum(p.numel() for p in trainable),
                                  "elapsed_seconds": time.time() - began}), flush=True)
                return
            if seen in CHECKPOINT_PAIRS:
                result = evaluate(model, processor, dev,
                                  image_index["development"],
                                  args.image_root / "development", dev_reference)
                summary = {key: value for key, value in result.items() if key != "rows"}
                summary.update({"pairs_seen": seen, "epoch": epoch + 1,
                                "elapsed_seconds": time.time() - began})
                checkpoints.append(summary)
                print(json.dumps(summary), flush=True)
                adapter = output / f"adapter_{seen}"
                model.save_pretrained(str(adapter), safe_serialization=True)
                (output / f"development_{seen}.json").write_text(
                    json.dumps(result, indent=2) + "\n")
                rank = (result["accuracy"], result["mean_margin_improvement"], -seen)
                if selected is None or rank > selected[0]:
                    selected = (rank, seen, summary)
    if seen != EPOCHS * len(fit) or seen != CHECKPOINT_PAIRS[-1] or selected is None:
        raise ValueError("incomplete preference training")
    baseline = checkpoints[0]["reference_accuracy"]
    gate = {"development_accuracy_at_least_0_60": selected[2]["accuracy"] >= .60,
            "development_gain_at_least_0_05": selected[2]["accuracy"] - baseline >= .05}
    report = {"schema": "group4_first_action_dpo_development_v1",
              "manifest_sha256": digest(args.manifest),
              "fit_image_index_sha256": digest(args.image_root / "fit" / "index.json"),
              "development_image_index_sha256": digest(args.image_root / "development" / "index.json"),
              "model_config_sha256": model_config_sha,
              "optimizer": {"beta": BETA, "learning_rate": LR,
                            "grad_accum": GRAD_ACCUM, "epochs": EPOCHS,
                            "checkpoints_pairs_seen": CHECKPOINT_PAIRS,
                            "episode_inverse_frequency_weighted": True},
              "fit_groups": len(fit), "fit_unique_episodes": len(counts),
              "development_groups": len(dev),
              "checkpoints": checkpoints,
              "selected_pairs_seen": selected[1], "gate": gate,
              "eligible_for_one_time_model_audit": all(gate.values()),
              "elapsed_seconds": time.time() - began}
    (output / "development_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_pairs_seen": selected[1],
                      "selected_development_accuracy": selected[2]["accuracy"],
                      "gate": gate, "elapsed_seconds": time.time() - began}), flush=True)


if __name__ == "__main__":
    main()
