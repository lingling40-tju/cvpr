"""Bounded instruction-conditioned progress reward for failed rollouts.

The environment invokes this server only after an unsuccessful trajectory.
Four sparse egocentric observations are encoded by the frozen navigation-SFT
model; a frozen failure-aware temporal head supplies a fit-calibrated score.
No simulator distance, goal position, or success flag enters the server.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache_navigation_sft_state import SYSTEM_PROMPT_R2R, USER_SUFFIX
from train_temporal_progress_encoder import TemporalPotential


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FailureOnlyReward:
    def __init__(self, args: argparse.Namespace) -> None:
        torch.set_num_threads(8)
        self.calibration = json.loads(args.calibration.read_text())
        if self.calibration["schema"] != "failure_rank_reward_calibration_v1" or \
                self.calibration["new_encoder_sha256"] != digest(args.temporal_checkpoint):
            raise ValueError("failure-rank encoder/calibration mismatch")
        self.temporal_scale = float(self.calibration["temporal_scale"])
        if not math.isfinite(self.temporal_scale) or self.temporal_scale <= .01:
            raise ValueError("invalid fit-only temporal scale")
        checkpoint = torch.load(args.temporal_checkpoint, map_location="cpu",
                                weights_only=False)
        if checkpoint["manifest_sha256"] != \
                self.calibration["failure_rank_manifest_sha256"]:
            raise ValueError("failure-rank manifest mismatch")
        self.temporal = TemporalPotential().to("cuda:0").eval()
        self.temporal.load_state_dict(checkpoint["model"])
        self.nav_processor = AutoProcessor.from_pretrained(
            str(args.nav_model), local_files_only=True, use_fast=False)
        if digest(args.nav_model / "config.json") != args.expected_nav_config_sha256:
            raise ValueError("navigation-SFT backbone mismatch")
        self.nav = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(args.nav_model), local_files_only=True,
            torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        self.lock = threading.Lock()
        self.requests = 0

    @torch.inference_mode()
    def score(self, payload: dict) -> dict:
        instruction = payload.get("instruction")
        image_b64 = payload.get("images")
        initial = payload.get("initial")
        if not isinstance(instruction, str) or not instruction.strip() or \
                not isinstance(image_b64, list) or len(image_b64) != 4 or \
                not isinstance(initial, list) or len(initial) != 4 or \
                not all(isinstance(flag, bool) for flag in initial) or not initial[0]:
            raise ValueError("invalid four-view scoring request")
        images = []
        for encoded in image_b64:
            if not isinstance(encoded, str) or len(encoded) > 2_000_000:
                raise ValueError("invalid image payload")
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as source:
                image = source.convert("RGB")
            image.thumbnail((336, 336))
            images.append(image)
        with self.lock:
            hidden = []
            for image, is_initial in zip(images, initial):
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R}]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "[Initial Observation]:\n" if is_initial
                         else "After that, the observation is:\n"},
                        {"type": "image", "image": image},
                        {"type": "text", "text": "\nInstruction: " + instruction + USER_SUFFIX},
                    ]},
                ]
                inputs = self.nav_processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to("cuda:0")
                output = self.nav(**inputs, output_hidden_states=True,
                                  use_cache=False)
                hidden.append(output.hidden_states[-1][0, -1].float())
                del output, inputs
            sequence = F.normalize(torch.stack(hidden).float(), dim=-1).unsqueeze(0)
            temporal_score = float(self.temporal(sequence)[0, -1])
            raw = temporal_score / self.temporal_scale
            bonus = 1 / (1 + math.exp(-max(-30.0, min(30.0, raw))))
            self.requests += 1
        if not all(math.isfinite(value) for value in (temporal_score, raw, bonus)):
            raise ValueError("nonfinite failure-only reward")
        return {"status": "ok", "temporal_score": temporal_score,
                "raw": raw, "bonus": bonus, "scored_views": 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8024)
    parser.add_argument("--nav-model", type=Path, required=True)
    parser.add_argument("--expected-nav-config-sha256", required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    scorer = FailureOnlyReward(args)

    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, value: dict) -> None:
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._write(200, {"status": "ok", "requests": scorer.requests,
                                  "reward_variant": "failure_only_temporal_v1",
                                  "calibration_sha256": digest(args.calibration),
                                  "encoder_sha256": digest(args.temporal_checkpoint)})
            else:
                self._write(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                self._write(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8_000_000:
                    raise ValueError("invalid content length")
                value = scorer.score(json.loads(self.rfile.read(length)))
                self._write(200, value)
            except Exception as exc:
                print(f"score error: {type(exc).__name__}: {exc}", flush=True)
                self._write(400, {"status": "error", "reason": str(exc)[:200]})

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"failure-only reward service ready port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
