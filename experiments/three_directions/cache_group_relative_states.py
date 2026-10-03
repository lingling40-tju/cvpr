"""Cache policy-history states at fixed turn anchors for complete group-four comparisons.

Fit and development are cached before model selection. Audit scenes may
be cached after a candidate passes its predeclared development gate.
Simulator distances are never passed to the model. The LoRA encoder was
frozen before this dataset was selected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model, rid


ANCHORS = (3, 6, 9, 12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--collection-audit", type=Path, required=True)
    parser.add_argument("--old-policy-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.collection_audit.read_text())
    manifest_sha = digest(args.manifest)
    if (manifest["schema"] != "policy_group_relative_manifest_v1" or
            audit["manifest_sha256"] != manifest_sha or
            not audit["group_comparison_sample_gate"]["passed"]):
        raise ValueError("group-four turn collection not audited")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint["schema"] not in ("history_grounding_lora_seed11_v1",
                                      "history_grounding_lora_interim_v1") or
            checkpoint["source_sha256"]["policy_manifest"] !=
                digest(args.old_policy_manifest) or
            checkpoint["model_config_sha256"] != digest(args.model / "config.json")):
        raise ValueError("frozen LoRA/source mismatch")
    source_id = digest(args.checkpoint)
    records = sorted(manifest["selected"][args.part], key=rid)
    records = records[args.shard::args.shards]
    if not records:
        raise ValueError("empty shard")
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    output = args.output_root / args.part
    (output / "records").mkdir(parents=True, exist_ok=True)
    completed = 0
    states = 0
    started = time.time()
    for number, plan in enumerate(records, 1):
        record_id = rid(plan)
        record = json.loads((args.turn_root / args.part / "records" /
                             f"{record_id}.json").read_text())
        if record["record_id"] != record_id or \
                record["manifest_sha256"] != manifest_sha:
            raise ValueError(f"invalid turn record {record_id}")
        indices = [index for index, turn in enumerate(record["turns"], 1)
                   if turn["original_turn_index"] in ANCHORS]
        anchor_turns = [record["turns"][index - 1]["original_turn_index"]
                        for index in indices]
        path = output / "records" / f"{record_id}.pt"
        if path.is_file():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            if (previous["source_id"] == source_id and
                    previous["manifest_sha256"] == manifest_sha and
                    previous["record_id"] == record_id and
                    previous["indices"] == indices and
                    previous["anchor_turns"] == anchor_turns and
                    previous["hidden"].shape ==
                        (len(indices), model.config.hidden_size) and
                    torch.isfinite(previous["hidden"]).all()):
                completed += 1
                states += len(indices)
                continue
        vectors = []
        item = {"record": record, "root": args.turn_root / args.part}
        with torch.inference_mode():
            for index in indices:
                inputs = build_inputs(processor, item, index).to("cuda")
                output_model = model(**inputs, output_hidden_states=False,
                                     use_cache=False)
                vector = output_model.logits[0, -1].float().cpu()
                if not torch.isfinite(vector).all():
                    raise ValueError(f"nonfinite state {record_id}/{index}")
                vectors.append(vector.to(torch.float16))
                del output_model, inputs
        hidden = torch.stack(vectors) if vectors else \
                 torch.empty((0, model.config.hidden_size), dtype=torch.float16)
        payload = {"schema": "group_relative_state_cache_v1",
                   "record_id": record_id, "source_id": source_id,
                   "manifest_sha256": manifest_sha,
                   "indices": indices, "anchor_turns": anchor_turns,
                   "hidden": hidden}
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)
        completed += 1
        states += len(indices)
        if number % 20 == 0:
            print(f"{args.part} shard={args.shard}/{args.shards} "
                  f"{number}/{len(records)} states={states} "
                  f"elapsed_s={time.time()-started:.1f}", flush=True)
    summary = {"schema": "group_relative_state_collection_v1",
               "part": args.part, "shard": args.shard, "shards": args.shards,
               "requested": len(records), "completed": completed,
               "states": states, "source_id": source_id,
               "manifest_sha256": manifest_sha,
               "elapsed_seconds": time.time() - started}
    (output / f"summary_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
