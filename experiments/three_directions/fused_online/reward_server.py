"""Frozen terminal potential service for the group-four reward pilot.

One request supplies four sparse egocentric views and the route instruction.
The score uses the exact offline temporal and SigLIP representations, with
fit-scene endpoint-margin scales and a bounded sigmoid bonus. Requests are
serialized for predictable GPU memory. No route ground truth enters scoring.
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
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from cache_navigation_sft_state import SYSTEM_PROMPT_R2R, USER_SUFFIX
from train_temporal_progress_encoder import TemporalPotential


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class RewardModel:
    def __init__(self, args) -> None:
        torch.set_num_threads(8)
        self.report = json.loads(args.development_report.read_text())
        if self.report["schema"] != "equal_temporal_visual_reward_development_v3_online_parity":
            raise ValueError("wrong calibration report")
        self.temporal_scale = float(self.report["fit_mean_absolute_margin_scales"]
                                    ["temporal"]["endpoint"])
        self.visual_scale = float(self.report["fit_mean_absolute_margin_scales"]
                                  ["visual"]["endpoint"])
        if min(self.temporal_scale, self.visual_scale) <= 0:
            raise ValueError("invalid fit-only scales")
        temporal_checkpoint = torch.load(args.temporal_checkpoint, map_location="cpu",
                                         weights_only=False)
        if temporal_checkpoint["v2_manifest_sha256"] != self.report["v2_manifest_sha256"]:
            raise ValueError("temporal checkpoint mismatch")
        self.temporal = TemporalPotential().to("cuda:0").eval()
        self.temporal.load_state_dict(temporal_checkpoint["model"])
        self.nav_processor = AutoProcessor.from_pretrained(
            str(args.nav_model), local_files_only=True, use_fast=False)
        self.nav = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(args.nav_model), local_files_only=True,
            torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        if digest(args.nav_model / "config.json") != self.report["sft_model_config_sha256"]:
            raise ValueError("navigation model config mismatch")
        self.visual_processor = AutoProcessor.from_pretrained(
            str(args.siglip_model), local_files_only=True, use_fast=False)
        base = AutoModel.from_pretrained(str(args.siglip_model),
                                         local_files_only=True,
                                         torch_dtype=torch.bfloat16)
        self.visual = get_peft_model(base, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=["q_proj", "v_proj"])).cuda().eval()
        adapter = torch.load(args.siglip_adapter, map_location="cpu", weights_only=False)
        if adapter["model_config_sha256"] != digest(args.siglip_model / "config.json") or \
                adapter["v2_manifest_sha256"] != self.report["v2_manifest_sha256"] or \
                adapter.get("text_padding") != "fixed_max_length_64":
            raise ValueError("SigLIP model/adaptor mismatch")
        set_peft_model_state_dict(self.visual, adapter["adapter"])
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
                output = self.nav(**inputs, output_hidden_states=True, use_cache=False)
                hidden.append(output.hidden_states[-1][0, -1].float())
                del output, inputs
            sequence = F.normalize(torch.stack(hidden).float(), dim=-1).unsqueeze(0)
            temporal_score = float(self.temporal(sequence)[0, -1])
            image_input = self.visual_processor(images=[images[-1]],
                                                return_tensors="pt")
            text_input = self.visual_processor(text=[instruction],
                                               padding="max_length", max_length=64,
                                               truncation=True, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vision = self.visual.get_image_features(
                    pixel_values=image_input["pixel_values"].to("cuda:0"))
                text = self.visual.get_text_features(
                    **{key: value.to("cuda:0") for key, value in text_input.items()})
            visual_score = float((F.normalize(vision.float(), dim=-1) *
                                  F.normalize(text.float(), dim=-1)).sum(-1)[0])
            raw = .5 * (temporal_score / self.temporal_scale +
                         visual_score / self.visual_scale)
            bonus = 1 / (1 + math.exp(-max(-30.0, min(30.0, raw))))
            self.requests += 1
        if not all(math.isfinite(value) for value in
                   (temporal_score, visual_score, raw, bonus)):
            raise ValueError("nonfinite fused reward")
        return {"status": "ok", "temporal_score": temporal_score,
                "visual_score": visual_score, "raw": raw, "bonus": bonus,
                "scored_views": 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--nav-model", type=Path, required=True)
    parser.add_argument("--siglip-model", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--siglip-adapter", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    args = parser.parse_args()
    scorer = RewardModel(args)

    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, data: dict) -> None:
            body = json.dumps(data, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._write(200, {"status": "ok", "requests": scorer.requests,
                                  "development_report_sha256": digest(args.development_report)})
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
                payload = json.loads(self.rfile.read(length))
                answer = scorer.score(payload)
                self._write(200, answer)
            except Exception as exc:
                print(f"score error: {type(exc).__name__}: {exc}", flush=True)
                self._write(400, {"status": "error", "reason": str(exc)[:200]})

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fused reward service ready port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
