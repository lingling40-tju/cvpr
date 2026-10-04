"""Small request-boundary test; does not load a reward checkpoint."""

from __future__ import annotations

import base64
from collections import OrderedDict
import io
import json
import threading
from types import SimpleNamespace

from PIL import Image
import torch

import future_advantage_score_server as service
from future_advantage_score_server import Scorer, decoded_item


def encoded_image(fmt: str = "JPEG", size: tuple[int, int] = (160, 120)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, (18, 42, 66)).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def rejected(row: dict, reason: str) -> None:
    try:
        with decoded_item(row):
            pass
    except ValueError:
        return
    raise AssertionError(f"accepted {reason}")


def main() -> None:
    jpg = encoded_image()
    history = [{"turn": turn, "executed_actions": ["move forward 25cm"]}
               for turn in range(1, 7)]
    for anchor, turns in ((3, (0, 3)), (6, (0, 3, 6))):
        row = {"instruction": "Go to the red chair.", "anchor": anchor,
               "images": {str(turn): jpg for turn in turns},
               "history": history[:anchor]}
        with decoded_item(row) as (instruction, frames, actions, parsed_anchor):
            assert instruction == row["instruction"]
            assert set(frames) == set(turns)
            assert len(actions) == anchor and parsed_anchor == anchor
        contaminated = json.loads(json.dumps(row))
        contaminated["oracle_distance"] = 2.0
        rejected(contaminated, "privileged oracle field")
        contaminated = json.loads(json.dumps(row))
        contaminated["history"] = history
        if anchor == 3:
            rejected(contaminated, "future actions")
        contaminated = json.loads(json.dumps(row))
        contaminated["images"]["6" if anchor == 3 else "9"] = jpg
        rejected(contaminated, "future image")
        contaminated = json.loads(json.dumps(row))
        contaminated["images"]["0"] = "not-base64"
        rejected(contaminated, "invalid image encoding")
        contaminated = json.loads(json.dumps(row))
        contaminated["images"]["0"] = encoded_image("PNG")
        rejected(contaminated, "non-JPEG image")
        contaminated = json.loads(json.dumps(row))
        contaminated["images"]["0"] = encoded_image(size=(400, 300))
        rejected(contaminated, "image that skipped the offline thumbnail")
        contaminated = json.loads(json.dumps(row))
        contaminated["history"][0]["executed_actions"] = ["stop"]
        rejected(contaminated, "STOP action")
        contaminated = json.loads(json.dumps(row))
        contaminated["instruction"] = "x" * 4097
        rejected(contaminated, "oversize instruction")

    class Inputs(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class FakeModel:
        calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(logits=torch.tensor([[[float(self.calls)]]]))

    scorer = Scorer.__new__(Scorer)
    scorer.processor = object()
    scorer.model = FakeModel()
    scorer.head = lambda hidden: hidden.sum()
    scorer.lock = threading.Lock()
    scorer.cache = OrderedDict()
    scorer.checkpoint_sha256 = "synthetic-no-model"
    original = service.build_live_inputs
    service.build_live_inputs = lambda *args: Inputs()
    try:
        first = scorer.score([row, row])
        assert first["scores"] == [1., 1.]
        assert first["model_calls"] == first["cache_hits"] == 1
        again = scorer.score([row])
        assert again["scores"] == [1.]
        assert again["model_calls"] == 0 and again["cache_hits"] == 1
        changed = json.loads(json.dumps(row))
        changed["instruction"] = "Go to the blue chair."
        other = scorer.score([changed])
        assert other["scores"] == [2.]
        assert other["model_calls"] == 1 and other["cache_hits"] == 0
        assert scorer.model.calls == 2
    finally:
        service.build_live_inputs = original
    print(json.dumps({"schema": "future_advantage_score_boundary_test_v2",
                      "anchors": [3, 6],
                      "privileged_and_future_inputs_rejected": True,
                      "duplicate_prefix_model_call_skipped": True,
                      "reward_checkpoint_loaded": False}))


if __name__ == "__main__":
    main()
