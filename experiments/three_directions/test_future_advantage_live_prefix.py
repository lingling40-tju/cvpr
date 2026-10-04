"""Check that live RGB bytes match the actual Habitat replay JPEG writer."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from collect_policy_preference_frames import save_frame
from future_advantage_live_prefix import canonical_jpeg, score_item


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="future_advantage_live_") as tmp:
        root = Path(tmp)
        frames = {}
        for turn, (width, height) in ((0, (640, 480)),
                                      (3, (320, 240)),
                                      (6, (80, 80))):
            rgb = np.arange(width * height * 3, dtype=np.uint32).reshape(
                height, width, 3).astype(np.uint8)
            frame_path = root / f"{turn}.jpg"
            save_frame({"rgb": rgb}, frame_path)
            with Image.fromarray(rgb) as frame:
                encoded = canonical_jpeg(frame)
            assert base64.b64decode(encoded) == frame_path.read_bytes()
            frames[turn] = rgb
        history = [{"turn": turn,
                    "executed_actions": ["move forward 25cm"]}
                   for turn in range(1, 7)]
        for anchor, needed in ((3, {0, 3}), (6, {0, 3, 6})):
            item = score_item("Go to the red chair.",
                              {turn: frames[turn] for turn in needed},
                              history[:anchor], anchor)
            assert set(item) == {"instruction", "anchor", "images", "history"}
            assert set(item["images"]) == {str(turn) for turn in needed}
            assert all(base64.b64decode(item["images"][str(turn)]) ==
                       (root / f"{turn}.jpg").read_bytes() for turn in needed)
        for frame_set, actions in (({0: frames[0], 3: frames[3], 6: frames[6]},
                                    history[:3]),
                                   ({0: frames[0], 3: frames[3]}, history)):
            try:
                score_item("Go to the red chair.", frame_set, actions, 3)
            except ValueError:
                pass
            else:
                raise AssertionError("accepted future RGB or actions")
        history[2]["executed_actions"] = ["stop"]
        try:
            score_item("Go to the red chair.",
                       {0: frames[0], 3: frames[3]}, history[:3], 3)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted STOP in active prefix")
    print(json.dumps({"schema": "future_advantage_live_jpeg_parity_v1",
                      "tested_shapes": [[640, 480], [320, 240], [80, 80]],
                      "byte_identical_to_replay_writer": True,
                      "future_and_stop_rejected": True}))


if __name__ == "__main__":
    main()
