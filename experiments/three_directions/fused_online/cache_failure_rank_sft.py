"""Cache navigation-SFT states for turn-sampled failed trajectory pairs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache_navigation_sft_state import PROMPT_VERSION, digest, encode_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != "failure_rank_train_scene_v1":
        raise ValueError("wrong failure-rank manifest")
    manifest_hash = digest(args.manifest)
    ids = [pair["pair_id"] + "_" + role for pair in manifest["pairs"]
           for role in ("near", "far")]
    if args.limit_records:
        if not 0 < args.limit_records <= len(ids):
            raise ValueError("invalid record limit")
        ids = ids[:args.limit_records]
    else:
        summary = json.loads((args.records_root / "summary.json").read_text())
        if summary["pairs"] != len(manifest["pairs"]) or \
                summary["requested_trajectories"] != len(ids) or \
                summary["completed_trajectories"] != len(ids) or \
                summary["errors"] or summary["manifest_sha256"] != manifest_hash:
            raise ValueError("full failure-rank replay incomplete")
    output = args.output_root
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    config_hash = digest(args.model / "config.json")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    token_ids = []
    for name in ("stop", "move", "turn"):
        encoded = processor.tokenizer.encode(name, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"action prefix {name} is not one token")
        token_ids.append(encoded[0])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    start = time.time()
    vectors, margins = [], []
    for number, record_id in enumerate(ids, 1):
        record = json.loads((args.records_root / "records" /
                             f"{record_id}.json").read_text())
        if record["record_id"] != record_id or \
                record["manifest_sha256"] != manifest_hash or \
                len(record["frames"]) != 4 or not record["frames"][0]["initial"]:
            raise ValueError(f"bad replay record {record_id}")
        cache_path = records_dir / f"{record_id}.pt"
        cached = None
        if cache_path.is_file():
            candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
            if candidate.get("record_id") == record_id and \
                    candidate.get("manifest_sha256") == manifest_hash and \
                    candidate.get("model_config_sha256") == config_hash and \
                    candidate.get("prompt_version") == PROMPT_VERSION and \
                    candidate["hidden"].shape == (4, model.config.hidden_size) and \
                    candidate["stop_margin"].shape == (4,) and \
                    torch.isfinite(candidate["hidden"]).all() and \
                    torch.isfinite(candidate["stop_margin"]).all():
                cached = candidate
        if cached is None:
            hidden, stop_margin = [], []
            for frame in record["frames"]:
                vector, margin = encode_one(
                    model, processor, args.records_root / frame["image"],
                    record["instruction"], tuple(token_ids),
                    initial=bool(frame["initial"]))
                hidden.append(vector)
                stop_margin.append(margin)
            cached = {"record_id": record_id, "manifest_sha256": manifest_hash,
                      "model_config_sha256": config_hash,
                      "prompt_version": PROMPT_VERSION,
                      "hidden": torch.stack(hidden),
                      "stop_margin": torch.tensor(stop_margin, dtype=torch.float32)}
            temporary = cache_path.with_suffix(".tmp")
            torch.save(cached, temporary)
            os.replace(temporary, cache_path)
        vectors.append(cached["hidden"])
        margins.append(cached["stop_margin"])
        if number % 20 == 0 or number == len(ids):
            print(f"states {number}/{len(ids)} elapsed_s={time.time()-start:.1f}",
                  flush=True)
    payload = {"schema": "failure_rank_navigation_sft_state_v1",
               "record_ids": ids, "manifest_sha256": manifest_hash,
               "model_config_sha256": config_hash,
               "prompt_version": PROMPT_VERSION,
               "sampling": "four evenly spaced initial/turn-boundary views",
               "hidden": torch.stack(vectors),
               "stop_margin": torch.stack(margins)}
    torch.save(payload, output / "features.pt")
    summary = {"records": len(ids), "frames": 4 * len(ids),
               "manifest_sha256": manifest_hash,
               "model_config_sha256": config_hash,
               "prompt_version": PROMPT_VERSION,
               "elapsed_seconds": time.time() - start,
               "output": str(output / "features.pt")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
