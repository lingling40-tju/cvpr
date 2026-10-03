"""Cache paired same-start expert instruction states at nonterminal prefixes.

Use the frozen navigation LoRA encoder and previously audited RGB histories.
The prefix counts are fixed to the policy outcome anchors, 3 and 6.
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
from fit_group4_future_success_linear import ANCHORS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), required=True)
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
        raise ValueError("empty expert prefix shard")
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    output = args.output_root / args.part
    (output / "records").mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = reused = encoded = 0
    max_context = 0
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
        anchors = [anchor for anchor in ANCHORS if anchor < record["turn_count"]]
        path = output / "records" / f"{eid}.pt"
        if path.is_file():
            prior = torch.load(path, map_location="cpu", weights_only=True)
            if prior["schema"] != "group4_expert_prefix_state_v1" or \
                    prior["episode_id"] != eid or prior["part"] != args.part or \
                    prior["manifest_sha256"] != manifest_sha or \
                    prior["record_sha256"] != row["record_sha256"] or \
                    prior["source_id"] != source_id or \
                    prior["anchors"] != anchors or \
                    prior["correct"].shape != (len(anchors), 2048) or \
                    prior["wrong"].shape != (len(anchors), 2048) or \
                    not bool(torch.isfinite(prior["correct"]).all()) or \
                    not bool(torch.isfinite(prior["wrong"]).all()):
                raise ValueError(f"bad reused expert prefix cache {eid}")
            completed += 1
            reused += 1
            continue
        item = {"record": record, "root": args.expert_root / row["source_part"]}
        correct, wrong = [], []
        with torch.inference_mode():
            for anchor in anchors:
                for instruction, target in ((record["instruction"], correct),
                                            (record["wrong_instruction"], wrong)):
                    inputs = build_inputs(processor, item, anchor,
                                          instruction).to("cuda")
                    max_context = max(max_context, int(inputs["input_ids"].shape[1]))
                    hidden = model(**inputs, output_hidden_states=False,
                                   use_cache=False).logits[0, -1].float().cpu()
                    if hidden.shape != (2048,) or \
                            not bool(torch.isfinite(hidden).all()):
                        raise ValueError(f"nonfinite expert prefix state {eid}/{anchor}")
                    target.append(hidden.to(torch.float16))
        payload = {"schema": "group4_expert_prefix_state_v1",
                   "episode_id": eid, "part": args.part,
                   "manifest_sha256": manifest_sha,
                   "record_sha256": row["record_sha256"],
                   "source_id": source_id,
                   "anchors": anchors,
                   "correct": torch.stack(correct) if correct else
                              torch.empty((0, 2048), dtype=torch.float16),
                   "wrong": torch.stack(wrong) if wrong else
                            torch.empty((0, 2048), dtype=torch.float16)}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        completed += 1
        encoded += 2 * len(anchors)
        if number % 20 == 0:
            print(f"{args.part} shard {args.shard}/{args.shards}: "
                  f"{number}/{len(rows)} elapsed_s={time.time()-started:.1f}",
                  flush=True)
    summary = {"schema": "group4_expert_prefix_state_shard_v1",
               "part": args.part, "shard": args.shard, "shards": args.shards,
               "manifest_sha256": manifest_sha, "source_id": source_id,
               "requested": len(rows), "completed": completed, "reused": reused,
               "states_encoded": encoded, "max_context_tokens": max_context,
               "elapsed_seconds": time.time() - started}
    (output / f"summary_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    if completed != len(rows):
        raise ValueError("incomplete expert prefix shard")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
