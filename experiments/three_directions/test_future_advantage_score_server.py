"""Small request-boundary test; does not load a reward checkpoint."""

from __future__ import annotations

import base64
import io
import json

from PIL import Image

from future_advantage_score_server import decoded_item


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
    print(json.dumps({"schema": "future_advantage_score_boundary_test_v1",
                      "anchors": [3, 6],
                      "privileged_and_future_inputs_rejected": True,
                      "reward_checkpoint_loaded": False}))


if __name__ == "__main__":
    main()
