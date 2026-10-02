"""Cache wrong-instruction SFT states for fixed temporal-v2 policy paths.

Each record reuses the existing four RGB frames of a successful group-four
policy trajectory. Fit and development are computed first; audit images
remain untouched until the new model passes development gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache_navigation_sft_state import encode_one


PROMPT_VERSION = "vlnce_server_single_observation_v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "development", "audit"), required=True)
    parser.add_argument("--limit-pairs", type=int, default=0)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    v2 = json.loads(args.v2_manifest.read_text())
    if v2["source_manifest_sha256"] != digest(args.manifest) or \
            v2["group_size"] != 4:
        raise ValueError("v2/source mismatch")
    pairs = {row["pair_id"]: row for row in source["pairs"]}
    selected = [row for row in v2["pairs"] if row["split"] == args.split]
    selected = selected[:args.limit_pairs or None]
    if not selected or args.limit_pairs < 0:
        raise ValueError("empty/invalid selection")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "records").mkdir(exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    action_tokens = []
    for word in ("stop", "move", "turn"):
        ids = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("invalid action prefix tokenizer")
        action_tokens.append(ids[0])
    model_hash = digest(args.model / "config.json")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    manifest_hash = digest(args.v2_manifest)
    begun = time.time()
    for count, item in enumerate(selected, 1):
        pair = pairs[item["pair_id"]]
        if pair["scene_id"] != item["scene_id"] or \
                pair["episode_id"] != item["source_episode_id"]:
            raise ValueError("pair identity mismatch")
        record_id = item["pair_id"] + "_success"
        record = json.loads((args.records_root / "records" /
                             f"{record_id}.json").read_text())
        if record["record_id"] != record_id or len(record["frames"]) != 4:
            raise ValueError(f"bad RGB record {record_id}")
        out = args.output_root / "records" / f"{item['pair_id']}.pt"
        cached = None
        if out.is_file():
            candidate = torch.load(out, map_location="cpu", weights_only=False)
            if candidate["pair_id"] == item["pair_id"] and \
                    candidate["manifest_sha256"] == manifest_hash and \
                    candidate["model_config_sha256"] == model_hash and \
                    candidate["prompt_version"] == PROMPT_VERSION and \
                    candidate["hidden"].shape == (4, 2048) and \
                    torch.isfinite(candidate["hidden"]).all():
                cached = candidate
        if cached is None:
            vectors = []
            for frame in record["frames"]:
                vector, _ = encode_one(
                    model, processor, args.records_root / frame["image"],
                    item["swapped_instruction"], tuple(action_tokens),
                    initial=frame["action_index"] == 0)
                vectors.append(vector)
            cached = {"pair_id": item["pair_id"], "split": item["split"],
                      "scene_id": item["scene_id"],
                      "manifest_sha256": manifest_hash,
                      "model_config_sha256": model_hash,
                      "prompt_version": PROMPT_VERSION,
                      "hidden": torch.stack(vectors)}
            temp = out.with_suffix(".tmp")
            torch.save(cached, temp)
            os.replace(temp, out)
        if count % 20 == 0 or count == len(selected):
            print(f"wrong-instruction {args.split} {count}/{len(selected)} "
                  f"elapsed_s={time.time()-begun:.1f}", flush=True)
    summary = {"schema": "temporal_contrastive_swap_cache_v1",
               "split": args.split, "records": len(selected),
               "v2_manifest_sha256": manifest_hash,
               "model_config_sha256": model_hash,
               "prompt_version": PROMPT_VERSION}
    (args.output_root / f"summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
