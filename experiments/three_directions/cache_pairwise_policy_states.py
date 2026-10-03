"""Cache selected policy-turn states for local, signed progress learning.

Only fit/development scenes are accessible here. One cached state can
serve multiple adjacent pairs; the same checkpoint is never recomputed
when training or evaluating several small pairwise heads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model, load_part


def state_indices(item: dict) -> list[int]:
    pairs = item["pairs"]
    afters = set(pairs["backward"])
    afters.update(pairs["forward"][:2])
    return sorted({index for after in afters for index in (after - 1, after)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-split", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-labels", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--policy-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.limit < 0:
        raise ValueError("invalid shard or limit")
    data = load_part(args.part, args.scene_split, args.expert_manifest,
                     args.expert_labels, args.expert_root,
                     args.policy_manifest, args.policy_root, args.policy_audit)
    selected = sorted(data["policy"], key=lambda item:
                      item["record"]["record_id"])
    selected = [item for item in selected if state_indices(item)]
    selected = [item for index, item in enumerate(selected)
                if index % args.shards == args.shard]
    selected = selected[:args.limit or None]
    if not selected:
        raise ValueError("empty shard")
    source_id = digest(args.checkpoint) if args.checkpoint else "base_sft"
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["schema"] != "history_grounding_lora_seed11_v1" or \
                checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
                checkpoint["source_sha256"]["policy_manifest"] != \
                    data["policy_manifest_sha256"]:
            raise ValueError("adapter/source mismatch")
    processor, model, _, _ = load_model(args.model)
    if checkpoint is not None:
        set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    output = args.output_root / args.part
    (output / "records").mkdir(parents=True, exist_ok=True)
    done = []
    started = time.time()
    for number, item in enumerate(selected, 1):
        record = item["record"]
        rid = record["record_id"]
        indices = state_indices(item)
        path = output / "records" / f"{rid}.pt"
        if path.is_file():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            if previous["source_id"] == source_id and \
                    previous["record_id"] == rid and \
                    previous["indices"] == indices and \
                    previous["policy_manifest_sha256"] == \
                        data["policy_manifest_sha256"] and \
                    previous["hidden"].shape == (len(indices), model.config.hidden_size) and \
                    torch.isfinite(previous["hidden"]).all():
                done.append(rid)
                continue
        vectors = []
        with torch.inference_mode():
            for index in indices:
                inputs = build_inputs(processor, item, index).to("cuda")
                output_model = model(**inputs, output_hidden_states=False,
                                     use_cache=False)
                vector = output_model.logits[0, -1].float().cpu()
                if not torch.isfinite(vector).all():
                    raise ValueError(f"nonfinite state {rid}/{index}")
                vectors.append(vector.to(torch.float16))
                del output_model, inputs
        payload = {"schema": "pairwise_policy_state_cache_v1",
                   "record_id": rid, "source_id": source_id,
                   "policy_manifest_sha256": data["policy_manifest_sha256"],
                   "indices": indices, "hidden": torch.stack(vectors)}
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)
        done.append(rid)
        if number % 20 == 0:
            print(f"{args.part} shard={args.shard}/{args.shards} "
                  f"{number}/{len(selected)} elapsed_s={time.time()-started:.1f}",
                  flush=True)
    summary = {"schema": "pairwise_policy_state_collection_v1",
               "part": args.part, "shard": args.shard,
               "shards": args.shards, "requested": len(selected),
               "completed": len(done), "source_id": source_id,
               "policy_manifest_sha256": data["policy_manifest_sha256"],
               "limit": args.limit, "elapsed_seconds": time.time() - started}
    (output / f"summary_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
