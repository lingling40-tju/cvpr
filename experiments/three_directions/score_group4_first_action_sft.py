"""Score frozen group-four first actions under the navigation SFT model.

This is an offline diagnostic, not a navigation evaluation. Each pair uses
one identical instruction and initial RGB; only the two assistant responses
differ. No terminal label enters the model prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from history_grounding_lora import build_inputs, digest


def shard(group_id: str, count: int) -> int:
    return int(hashlib.sha256(group_id.encode()).hexdigest(), 16) % count


@torch.inference_mode()
def logprob(model, processor, image_root: Path, image: dict,
            response: str) -> dict:
    item = {"root": image_root, "record": {
        "instruction": image["instruction"], "initial_image": image["image"],
        "turns": []}}
    prompt = build_inputs(processor, item, 0).to("cuda")
    prompt_len = int(prompt["input_ids"].shape[1])
    action_ids = processor.tokenizer.encode(response, add_special_tokens=False)
    if not action_ids or processor.tokenizer.eos_token_id is None:
        raise ValueError("empty response or missing EOS")
    suffix = action_ids + [processor.tokenizer.eos_token_id]
    suffix_tensor = torch.tensor([suffix], dtype=prompt["input_ids"].dtype,
                                 device="cuda")
    full = dict(prompt)
    full["input_ids"] = torch.cat((prompt["input_ids"], suffix_tensor), dim=1)
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    logits = model(**full, use_cache=False).logits[0,
        prompt_len - 1:prompt_len + len(suffix) - 1].float()
    targets = suffix_tensor[0]
    selected = F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("nonfinite response log probabilities")
    action = selected[:-1]
    return {"action_token_count": len(action_ids),
            "action_logprob_sum": float(action.sum().cpu()),
            "action_logprob_mean": float(action.mean().cpu()),
            "with_eos_logprob_sum": float(selected.sum().cpu()),
            "prompt_tokens": prompt_len}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.limit < 0:
        raise ValueError("bad shard")
    manifest = json.loads(args.manifest.read_text())
    index_path = args.image_root / args.part / "index.json"
    index = json.loads(index_path.read_text())
    if manifest["schema"] != "group4_first_action_preference_manifest_v1" or \
            index["schema"] != "group4_first_action_image_index_v1" or \
            index["manifest_sha256"] != digest(args.manifest):
        raise ValueError("manifest/image index mismatch")
    rows = [row for row in manifest["selected"][args.part]
            if shard(row["group_id"], args.shards) == args.shard]
    rows = rows[:args.limit or None]
    if not rows:
        raise ValueError("empty model score shard")
    processor = AutoProcessor.from_pretrained(str(args.model),
                                               local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        local_files_only=True).cuda().eval()
    scores = []
    for row in rows:
        eid = str(row["episode_id"])
        image = index["images"][eid]
        path = args.image_root / args.part / image["image"]
        if image["scene_id"] != row["scene_id"] or \
                image["instruction"] != row["instruction"] or \
                digest(path) != image["image_sha256"]:
            raise ValueError(f"image mismatch for {row['group_id']}")
        good = logprob(model, processor, args.image_root / args.part,
                       image, row["preferred_response"])
        bad = logprob(model, processor, args.image_root / args.part,
                      image, row["rejected_response"])
        scores.append({"group_id": row["group_id"], "episode_id": eid,
                       "scene_id": row["scene_id"],
                       "preferred": good, "rejected": bad})
        if len(scores) % 10 == 0:
            print(f"{args.part} shard {args.shard}: {len(scores)}/{len(rows)}", flush=True)
    metrics = {}
    for key in ("action_logprob_mean", "action_logprob_sum",
                "with_eos_logprob_sum"):
        deltas = [row["preferred"][key] - row["rejected"][key]
                  for row in scores]
        metrics[key] = {"correct": sum(delta > 0 for delta in deltas),
                        "ties": sum(delta == 0 for delta in deltas),
                        "pairs": len(deltas),
                        "accuracy": sum(delta > 0 for delta in deltas) / len(deltas)}
    result = {"schema": "group4_first_action_sft_score_v1", "part": args.part,
              "shard": args.shard, "shards": args.shards,
              "smoke_limit": args.limit,
              "manifest_sha256": digest(args.manifest),
              "image_index_sha256": digest(index_path),
              "model_config_sha256": digest(args.model / "config.json"),
              "method": "teacher-forced Qwen2.5-VL-3B navigation-SFT action response logprob under original first-turn prompt",
              "metrics": metrics, "rows": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
