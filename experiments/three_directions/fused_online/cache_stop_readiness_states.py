"""Cache frozen navigation-SFT states for the two-frame STOP probe.

All serialized tensor loads use PyTorch's restricted weights-only mode.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache_navigation_sft_state import PROMPT_VERSION, encode_one


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-records", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert manifest["schema"] == "stop_readiness_onpolicy_train_v1"
    summary = json.loads((args.records_root / "summary.json").read_text())
    assert summary["manifest_sha256"] == digest(args.manifest)
    assert summary["completed_trajectories"] == summary["requested_trajectories"]
    assert not summary["errors"]
    ids = [row["record_id"] for row in manifest["records"]]
    if args.limit_records:
        assert 0 < args.limit_records <= len(ids)
        ids = ids[:args.limit_records]
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_root / "records"
    cache_dir.mkdir(exist_ok=True)
    manifest_hash = digest(args.manifest)
    config_hash = digest(args.model / "config.json")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              use_fast=False)
    token_ids = []
    for name in ("stop", "move", "turn"):
        encoded = processor.tokenizer.encode(name, add_special_tokens=False)
        assert len(encoded) == 1
        token_ids.append(encoded[0])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        device_map="cuda:0", local_files_only=True).eval()
    hidden = []
    margins = []
    started = time.time()
    for number, record_id in enumerate(ids, 1):
        source = json.loads((args.records_root / "records" / f"{record_id}.json").read_text())
        assert source["record_id"] == record_id and len(source["frames"]) == 2
        path = cache_dir / f"{record_id}.pt"
        cached = None
        if path.is_file():
            candidate = torch.load(path, map_location="cpu", weights_only=True)
            if candidate.get("record_id") == record_id and \
                    candidate.get("manifest_sha256") == manifest_hash and \
                    candidate.get("model_config_sha256") == config_hash and \
                    candidate["hidden"].shape == (2, model.config.hidden_size) and \
                    candidate["stop_margin"].shape == (2,) and \
                    torch.isfinite(candidate["hidden"]).all() and \
                    torch.isfinite(candidate["stop_margin"]).all():
                cached = candidate
        if cached is None:
            vectors = []
            scores = []
            for frame in source["frames"]:
                vector, margin = encode_one(
                    model, processor, args.records_root / frame["image"],
                    source["instruction"], tuple(token_ids),
                    initial=frame["action_index"] == 0,
                )
                vectors.append(vector)
                scores.append(margin)
            cached = {
                "record_id": record_id,
                "manifest_sha256": manifest_hash,
                "model_config_sha256": config_hash,
                "prompt_version": PROMPT_VERSION,
                "hidden": torch.stack(vectors),
                "stop_margin": torch.tensor(scores, dtype=torch.float32),
            }
            temp = path.with_suffix(".tmp")
            torch.save(cached, temp)
            os.replace(temp, path)
        hidden.append(cached["hidden"])
        margins.append(cached["stop_margin"])
        if number % 20 == 0 or number == len(ids):
            print(f"states {number}/{len(ids)} elapsed_s={time.time()-started:.1f}",
                  flush=True)
    payload = {
        "schema": "stop_readiness_onpolicy_sft_state_v1",
        "record_ids": ids,
        "manifest_sha256": manifest_hash,
        "model_config_sha256": config_hash,
        "prompt_version": PROMPT_VERSION,
        "hidden": torch.stack(hidden),
        "stop_margin": torch.stack(margins),
    }
    torch.save(payload, args.output_root / "features.pt")
    (args.output_root / "summary.json").write_text(json.dumps({
        "records": len(ids), "frames": 2 * len(ids),
        "manifest_sha256": manifest_hash,
        "model_config_sha256": config_hash,
        "elapsed_seconds": time.time() - started,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
