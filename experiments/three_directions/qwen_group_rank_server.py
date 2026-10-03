"""Frozen Qwen3-VL route margin for an exact-start n=4 reward pilot.

The server receives six causal RGB snapshots and the task instruction.
Its wrong instruction comes only from a frozen train-split mapping. It
never sees simulator distance, success, or the terminal mode. The
optimizer later uses raw scores solely within matched failure groups.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from score_qwen3_route_match import SYSTEM, INTRO, ENDING, logit_margin


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FrozenRouteReward:
    def __init__(self, args: argparse.Namespace) -> None:
        torch.set_num_threads(8)
        self.manifest = json.loads(args.manifest.read_text())
        self.expert = json.loads(args.expert_analysis.read_text())
        expected_rows = {
            "qwen3_group4_exact_start_dataset_v1": 256,
            "qwen3_group4_exact_start_scale_dataset_v1": 512,
        }.get(self.manifest["schema"])
        if expected_rows is None or \
                self.manifest["selected_rows"] != expected_rows or \
                self.manifest["source_sha256"]["train_dataset"] != \
                digest(args.train_dataset) or \
                self.manifest["output_parquet_sha256"] != digest(args.pilot_parquet) or \
                self.expert["schema"] != "qwen3_route_match_analysis_v1" or \
                not self.expert["gate"]["passed"] or \
                self.expert["source_sha256"]["model_hash_file"] != \
                digest(args.model_hashes) or \
                self.expert["source_sha256"]["model_config"] != \
                digest(args.model / "config.json") or \
                self.expert["source_sha256"]["tokenizer"] != \
                digest(args.model / "tokenizer.json") or \
                self.expert["source_sha256"]["prompt"] != \
                hashlib.sha256((SYSTEM + INTRO + ENDING).encode()).hexdigest():
            raise ValueError("frozen dataset, teacher, or prompt mismatch")
        with gzip.open(args.train_dataset, "rt", encoding="utf-8") as stream:
            episodes = json.load(stream)["episodes"]
        by_id = {str(row["episode_id"]): row for row in episodes}
        if len(by_id) != len(episodes):
            raise ValueError("duplicate train episode ID")
        self.instructions = {}
        for row in self.manifest["rows"]:
            eid, wrong_id = row["episode_id"], row["wrong_episode_id"]
            original, wrong = by_id[eid], by_id[wrong_id]
            self.instructions[eid] = (
                original["instruction"]["instruction_text"].strip(),
                wrong["instruction"]["instruction_text"].strip())
        if len(self.instructions) != expected_rows:
            raise ValueError("duplicate selected train ID")
        self.processor = AutoProcessor.from_pretrained(
            str(args.model), local_files_only=True)
        if self.processor.tokenizer.encode("A", add_special_tokens=False) != [32] or \
                self.processor.tokenizer.encode("B", add_special_tokens=False) != [33]:
            raise ValueError("A/B token IDs changed")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(args.model), dtype=torch.bfloat16, device_map="cuda:0",
            local_files_only=True).eval()
        self.lock = threading.Lock()
        self.requests = 0

    def score(self, payload: dict) -> dict:
        eid = str(payload.get("episode_id", ""))
        instructions = self.instructions.get(eid)
        if instructions is None or \
                str(payload.get("instruction", "")).strip() != instructions[0]:
            raise ValueError("episode/instruction not in frozen exact-start mapping")
        encoded = payload.get("images")
        if not isinstance(encoded, list) or len(encoded) != 6:
            raise ValueError("six ordered route views required")
        frames = []
        try:
            for item in encoded:
                if not isinstance(item, str) or len(item) > 2_000_000:
                    raise ValueError("invalid image payload")
                with Image.open(io.BytesIO(base64.b64decode(item, validate=True))) as source:
                    frame = source.convert("RGB")
                if max(frame.size) > 448:
                    frame.thumbnail((448, 448), Image.Resampling.BICUBIC)
                frames.append(frame)
            with self.lock:
                first, _, tokens_first = logit_margin(
                    self.model, self.processor, frames,
                    instructions[0], instructions[1], 32, 33)
                swapped, _, tokens_swapped = logit_margin(
                    self.model, self.processor, frames,
                    instructions[1], instructions[0], 32, 33)
                raw = (first - swapped) / 2
                if not math.isfinite(raw):
                    raise ValueError("nonfinite teacher margin")
                self.requests += 1
            return {"status": "ok", "raw": raw, "bonus": 0.0,
                    "first_margin": first, "swapped_margin": -swapped,
                    "input_tokens": [tokens_first, tokens_swapped],
                    "scored_views": 6, "episode_id": eid}
        finally:
            for frame in frames:
                frame.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-parquet", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-hashes", type=Path, required=True)
    parser.add_argument("--expert-analysis", type=Path, required=True)
    args = parser.parse_args()
    scorer = FrozenRouteReward(args)

    class Handler(BaseHTTPRequestHandler):
        def write(self, status: int, value: dict) -> None:
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.write(404, {"error": "not found"})
            else:
                self.write(200, {"status": "ok", "requests": scorer.requests,
                                 "variant": "qwen3_exact_start_group_rank_v1",
                                 "manifest_sha256": digest(args.manifest),
                                 "expert_analysis_sha256": digest(args.expert_analysis)})

        def do_POST(self):
            if self.path != "/score":
                self.write(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_000_000:
                    raise ValueError("invalid request length")
                self.write(200, scorer.score(json.loads(self.rfile.read(length))))
            except Exception as exc:
                print(f"teacher request error: {type(exc).__name__}: {exc}", flush=True)
                self.write(400, {"status": "error", "reason": str(exc)[:200]})

        def log_message(self, format, *args):
            return

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
