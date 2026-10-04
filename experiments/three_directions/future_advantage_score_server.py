"""Local observation-only Qwen prefix scorer for a future gated n=4 pilot.

Staged only. No server is started and no navigation inference is claimed by
this module's synthetic request checks. The launch path must separately
enforce the development, goal-swap, and prospective audit admissions.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager
import io
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

from PIL import Image
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from future_advantage_visual_input import build_live_inputs
from train_future_advantage_sparse_lora import digest, load_model


MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_PIXELS = 4_000_000
MAX_BATCH = 64
MAX_REQUEST_BYTES = 40_000_000


@contextmanager
def decoded_item(row: dict):
    """Reject every non-observation request field before model inference."""
    if not isinstance(row, dict) or set(row) != {
            "instruction", "anchor", "images", "history"}:
        raise ValueError("score item has privileged or missing fields")
    anchor = row["anchor"]
    needed = [0, 3] if anchor == 3 else [0, 3, 6] if anchor == 6 else None
    if needed is None or not isinstance(row["instruction"], str) or \
            not row["instruction"].strip() or \
            len(row["instruction"]) > 4096 or \
            not isinstance(row["images"], dict) or \
            set(row["images"]) != {str(turn) for turn in needed} or \
            not isinstance(row["history"], list):
        raise ValueError("invalid sparse score prefix")
    frames = {}
    try:
        for turn in needed:
            encoded = row["images"][str(turn)]
            if not isinstance(encoded, str) or len(encoded) > \
                    4 * MAX_IMAGE_BYTES // 3 + 8:
                raise ValueError("image encoding too large or missing")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds byte limit")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG" or \
                        image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("expected bounded JPEG observation")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                frames[turn] = image.convert("RGB")
        history = row["history"]
        if len(history) != anchor or not all(
                isinstance(turn, dict) for turn in history) or \
                [turn.get("turn") for turn in history] != \
                list(range(1, anchor + 1)):
            raise ValueError("future, missing, or unordered executed actions")
        for turn in history:
            if set(turn) != {"turn", "executed_actions"} or \
                    not isinstance(turn["executed_actions"], list) or \
                    not 1 <= len(turn["executed_actions"]) <= 16 or any(
                        not isinstance(action, str) or
                        not 0 < len(action) <= 80 or
                        "stop" in action.lower()
                        for action in turn["executed_actions"]):
                raise ValueError("invalid executed movement history")
        yield row["instruction"], frames, history, anchor
    finally:
        for frame in frames.values():
            frame.close()


class Scorer:
    def __init__(self, model_path: Path, checkpoint_path: Path):
        saved = torch.load(checkpoint_path, map_location="cpu",
                           weights_only=False)
        if saved.get("schema") != "future_advantage_sparse_lora_final_v1" or \
                saved.get("microsteps") != 1024 or \
                saved.get("accumulation") != 4 or \
                saved.get("model_config_sha256") != digest(
                    model_path / "config.json") or \
                saved.get("visual_input_sha256") != digest(
                    Path(__file__).with_name("future_advantage_visual_input.py")):
            raise ValueError("fixed reward checkpoint or visual input changed")
        self.processor, self.model, self.head = load_model(model_path)
        expected = get_peft_model_state_dict(self.model)
        if set(saved["adapter"]) != set(expected) or any(
                saved["adapter"][key].shape != expected[key].shape
                for key in expected):
            raise ValueError("LoRA checkpoint keys or shapes changed")
        set_peft_model_state_dict(self.model, saved["adapter"])
        self.head.load_state_dict(saved["head"], strict=True)
        self.model.eval()
        self.head.eval()
        self.lock = threading.Lock()
        self.checkpoint_sha256 = digest(checkpoint_path)

    def score(self, items: list[dict]) -> dict:
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH:
            raise ValueError("score batch must contain 1 to 64 prefixes")
        started = time.time()
        values = []
        with self.lock, torch.inference_mode():
            for row in items:
                with decoded_item(row) as (instruction, frames, history, anchor):
                    inputs = build_live_inputs(self.processor, instruction,
                                               frames, history, anchor).to("cuda")
                    output = self.model(**inputs, output_hidden_states=False,
                                        use_cache=False)
                    value = float(self.head(output.logits[0, -1]))
                    if not math.isfinite(value):
                        raise ValueError("nonfinite observation prefix score")
                    values.append(value)
        return {"scores": values,
                "checkpoint_sha256": self.checkpoint_sha256,
                "elapsed_seconds": time.time() - started}


def make_handler(scorer: Scorer):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, value: dict) -> None:
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"schema": "future_advantage_score_service_v1",
                                 "checkpoint_sha256": scorer.checkpoint_sha256})
            else:
                self._json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/score":
                self._json(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("invalid request byte count")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {"items"}:
                    raise ValueError("score request must contain only items")
                self._json(200, scorer.score(payload["items"]))
            except (ValueError, TypeError, KeyError, OSError,
                    binascii.Error) as error:
                self._json(400, {"error": str(error)[:160]})
            except Exception:
                self._json(500, {"error": "score inference failed"})

        def log_message(self, format, *args):
            # Never log navigation instructions or encoded RGB payloads.
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535:
        raise ValueError("score server must remain on a local TCP port")
    scorer = Scorer(args.model, args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scorer))
    print(json.dumps({"schema": "future_advantage_score_service_v1",
                      "checkpoint_sha256": scorer.checkpoint_sha256,
                      "port": args.port}), flush=True)
    server.serve_forever(poll_interval=.5)


if __name__ == "__main__":
    main()
