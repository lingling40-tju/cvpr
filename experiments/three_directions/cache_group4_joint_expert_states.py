"""Cache correct/wrong instruction states on existing expert RGB histories.

The navigation-SFT LoRA encoder is frozen. Each expert final history is
encoded exactly twice, with an audited natural same-start instruction
pair. No Habitat run or geodesic model input is needed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("bad shard")
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest_sha = digest(args.manifest)
    source_id = digest(args.checkpoint)
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1" or \
            checkpoint["schema"] not in ("history_grounding_lora_seed11_v1",
                                         "history_grounding_lora_interim_v1") or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
            checkpoint["source_sha256"]["expert_manifest"] != \
                manifest["source_sha256"]["expert_manifest"] or \
            checkpoint["source_sha256"]["scene_split"] != \
                manifest["source_sha256"]["scene_split"]:
        raise ValueError("frozen LoRA/source mismatch")
    rows = manifest["selected"][args.part][args.shard::args.shards]
    if not rows:
        raise ValueError("empty expert state shard")
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    output = args.output_root / args.part
    (output / "records").mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    reused = 0
    context_lengths = []
    for number, row in enumerate(rows, 1):
        eid = str(row["episode_id"])
        record_path = args.expert_root / row["source_part"] / "records" / f"{eid}.json"
        if digest(record_path) != row["record_sha256"]:
            raise ValueError(f"expert source hash mismatch {eid}")
        record = json.loads(record_path.read_text())
        if record["episode_id"] != eid or \
                record["scene_id"] != row["scene_id"] or \
                record["trajectory_id"] != row["trajectory_id"] or \
                record["turn_count"] != row["turn_count"] or \
                record["wrong_instruction_episode_id"] != row["wrong_episode_id"]:
            raise ValueError(f"expert record identity mismatch {eid}")
        path = output / "records" / f"{eid}.pt"
        if path.is_file():
            prior = torch.load(path, map_location="cpu", weights_only=True)
            if prior.get("manifest_sha256") == manifest_sha and \
                    prior.get("source_id") == source_id and \
                    prior.get("record_sha256") == row["record_sha256"] and \
                    prior.get("part") == args.part and \
                    prior["correct"].shape == (2048,) and \
                    prior["wrong"].shape == (2048,) and \
                    bool(torch.isfinite(prior["correct"]).all()) and \
                    bool(torch.isfinite(prior["wrong"]).all()):
                completed += 1
                reused += 1
                continue
        item = {"record": record, "root": args.expert_root / row["source_part"]}
        vectors = []
        with torch.inference_mode():
            for instruction in (record["instruction"], record["wrong_instruction"]):
                inputs = build_inputs(processor, item, record["turn_count"],
                                      instruction).to("cuda")
                context_lengths.append(int(inputs["input_ids"].shape[1]))
                hidden = model(**inputs, output_hidden_states=False,
                               use_cache=False).logits[0, -1].float().cpu()
                if not bool(torch.isfinite(hidden).all()):
                    raise ValueError(f"nonfinite expert state {eid}")
                vectors.append(hidden.to(torch.float16))
        payload = {"schema": "group4_joint_expert_state_v1",
                   "episode_id": eid, "part": args.part,
                   "manifest_sha256": manifest_sha,
                   "record_sha256": row["record_sha256"],
                   "source_id": source_id,
                   "correct": vectors[0], "wrong": vectors[1]}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        completed += 1
        if number % 20 == 0:
            print(f"{args.part} shard {args.shard}/{args.shards}: "
                  f"{number}/{len(rows)} elapsed_s={time.time()-started:.1f}",
                  flush=True)
    summary = {"schema": "group4_joint_expert_state_shard_v1",
               "part": args.part, "shard": args.shard, "shards": args.shards,
               "manifest_sha256": manifest_sha, "source_id": source_id,
               "requested": len(rows), "completed": completed, "reused": reused,
               "states_encoded": 2 * (completed - reused),
               "max_context_tokens": max(context_lengths, default=0),
               "elapsed_seconds": time.time() - started}
    (output / f"summary_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    if completed != len(rows):
        raise ValueError("incomplete expert state shard")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
