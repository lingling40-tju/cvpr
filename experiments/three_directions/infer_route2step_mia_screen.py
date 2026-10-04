"""Cache Route2Step MIA text on the frozen group-four development screen.

Private responses and licensed instruction data stay on the experiment host.
This produces no reward or navigation metric. A partial JSONL is resumable.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from transformers import (AutoProcessor, Qwen2_5_VLConfig,
                          Qwen2_5_VLForConditionalGeneration)

from smoke_route2step_mia_progress import (SYSTEM, IMAGE_TOKEN, digest,
                                           prompt, selected_frames)
from analyze_route2step_mia_screen import stage_score


def fail_if_gate_impossible(reasons: Counter, completed: int,
                            output: Path) -> None:
    # The frozen complete screen requires >=90% answer tags (at least 116
    # of 128) and >=80% aligned answers (at least 103 of 128). Once 13
    # tags or 26 alignments are lost, no remaining response can rescue it.
    missing_tags = reasons["missing_answer_tag"]
    unaligned = completed - reasons["aligned"]
    if missing_tags <= 12 and unaligned <= 25:
        return
    report = {
        "schema": "route2step_mia_format_early_failure_v1",
        "completed_queries": completed, "planned_queries": 128,
        "reason_counts": dict(reasons),
        "missing_answer_tags": missing_tags,
        "nonaligned_answers": unaligned,
        "max_possible_tag_fraction": (128 - missing_tags) / 128,
        "max_possible_alignment_fraction": (128 - unaligned) / 128,
        "interpretation": "Frozen format/alignment gate is mathematically impossible; no pairwise development score or navigation result",
    }
    path = output.with_name("early_failure.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    raise RuntimeError(f"MIA exploratory format gate cannot pass: {path}")


def load_model(root: Path):
    processor = AutoProcessor.from_pretrained(str(root), local_files_only=True,
                                              use_fast=False)
    config_data = json.loads((root / "config.json").read_text())
    nested = config_data.pop("text_config")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "intermediate_size", "vocab_size"):
        if config_data[key] != nested[key]:
            raise ValueError(f"checkpoint text architecture mismatch: {key}")
    config_data["tie_word_embeddings"] = nested["tie_word_embeddings"]
    config = Qwen2_5_VLConfig(**config_data)
    model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(root), config=config, torch_dtype=torch.bfloat16,
        local_files_only=True, output_loading_info=True)
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise ValueError(f"checkpoint weights did not load exactly: {loading}")
    return processor, model.cuda().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("model", "manifest", "record-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.manifest.read_text())
    if frozen["schema"] != "route2step_mia_group4_screen_manifest_v1" or \
            frozen["groups"] != 16 or frozen["queries"] != 128:
        raise ValueError("unexpected frozen screen")
    manifest_sha = digest(args.manifest)
    completed = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            key = (row["record_id"], row["anchor"])
            if key in completed or row["screen_manifest_sha256"] != manifest_sha:
                raise ValueError("duplicate or wrong-manifest cached query")
            completed[key] = row
    planned = []
    for group in frozen["selected"]:
        for rec in group["records"]:
            for anchor in frozen["anchors"]:
                planned.append((group, rec, anchor))
    if len(planned) != 128 or \
            set(completed) - {(r["record_id"], a) for _, r, a in planned}:
        raise ValueError("cache keys do not match frozen plan")
    plan_hashes = {(r["record_id"], a): r["sha256"]
                   for _, r, a in planned}
    reasons = Counter()
    for key, cached in completed.items():
        path = (args.record_root / "development" / "records" /
                f"{cached['record_id']}.json")
        if cached["record_sha256"] != plan_hashes[key] or \
                digest(path) != plan_hashes[key]:
            raise ValueError("cached response source changed")
        record = json.loads(path.read_text())
        _, reason = stage_score(record["instruction"], cached["response"])
        reasons[reason] += 1
    fail_if_gate_impossible(reasons, len(completed), args.output)
    if len(completed) == len(planned):
        print(json.dumps({"complete": len(completed), "new_queries": 0}))
        return
    processor, model = load_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    with args.output.open("a") as out:
        for group, rec, anchor in planned:
            rid = rec["record_id"]
            if (rid, anchor) in completed:
                continue
            path = (args.record_root / "development" / "records" /
                    f"{rid}.json")
            record = json.loads(path.read_text())
            if digest(path) != rec["sha256"] or \
                    record["record_id"] != rid or \
                    record["scene_id"] != group["scene_id"] or \
                    str(record["episode_id"]) != group["episode_id"]:
                raise ValueError(f"source record changed: {rid}")
            images, history_count, current_count = selected_frames(
                record, args.record_root / "development", anchor)
            try:
                query = prompt(record["instruction"], history_count,
                               current_count)
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": query}]
                rendered = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                rendered = rendered.replace("<image>", IMAGE_TOKEN)
                begin = time.perf_counter()
                inputs = processor(text=[rendered], images=images,
                                   return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    result = model.generate(**inputs, max_new_tokens=128,
                                            do_sample=False)
                response = processor.batch_decode(
                    result[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0]
                row = {"screen_manifest_sha256": manifest_sha,
                       "record_id": rid, "record_sha256": rec["sha256"],
                       "anchor": anchor, "response": response,
                       "input_tokens": int(inputs["input_ids"].shape[1]),
                       "output_tokens": int(result.shape[1] - inputs["input_ids"].shape[1]),
                       "elapsed_seconds": time.perf_counter() - begin}
                out.write(json.dumps(row) + "\n")
                out.flush()
                new_count += 1
                _, reason = stage_score(record["instruction"], response)
                reasons[reason] += 1
                fail_if_gate_impossible(reasons, len(completed) + new_count,
                                        args.output)
                if new_count % 8 == 0:
                    print(json.dumps({"new_queries": new_count,
                                      "complete_queries": len(completed) + new_count}),
                          flush=True)
            finally:
                for image in images:
                    image.close()
    print(json.dumps({"new_queries": new_count,
                      "complete_queries": len(completed) + new_count}))


if __name__ == "__main__":
    main()
