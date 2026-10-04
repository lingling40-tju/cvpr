"""Check offline/live sparse prompts with the actual Qwen2.5-VL processor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from PIL import Image
import torch
from transformers import AutoProcessor

from future_advantage_visual_input import build_inputs, build_live_inputs


def reject(call, label: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"accepted {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(str(args.model),
                                              local_files_only=True,
                                              use_fast=False)
    with tempfile.TemporaryDirectory(prefix="future_advantage_prompt_") as tmp:
        root = Path(tmp)
        colors = {0: (24, 36, 48), 3: (88, 67, 35), 6: (120, 49, 72)}
        paths = {}
        for turn, rgb in colors.items():
            path = root / f"{turn}.png"
            Image.new("RGB", (160, 120), rgb).save(path)
            paths[str(turn)] = path.name
        history = [{"turn": turn, "executed_actions": [
            "move forward 25cm" if turn % 2 else "turn left 15deg"]}
            for turn in range(1, 7)]
        record = {"schema": "future_advantage_sparse_model_input_v1",
                  "input": {"instruction": "Go to the red chair.",
                            "images": paths,
                            "action_history_by_anchor": {
                                "3": history[:3], "6": history}}}
        counts = {}
        for anchor in (3, 6):
            needed = (0, 3) if anchor == 3 else (0, 3, 6)
            with_frames = {}
            try:
                for turn in needed:
                    with_frames[turn] = Image.open(root / paths[str(turn)])
                disk = build_inputs(processor, record, root, anchor)
                live = build_live_inputs(processor, record["input"]["instruction"],
                                         with_frames, history[:anchor], anchor)
                if set(disk) != set(live) or any(
                        not torch.equal(disk[key], live[key]) for key in disk):
                    raise AssertionError(f"offline/live processor mismatch at {anchor}")
                counts[str(anchor)] = {"images": len(needed),
                                       "tokens": int(disk["input_ids"].shape[1])}
            finally:
                for frame in with_frames.values():
                    frame.close()
        with Image.open(root / "6.png") as extra:
            reject(lambda: build_live_inputs(
                processor, record["input"]["instruction"],
                {0: extra, 3: extra, 6: extra}, history[:3], 3),
                "future turn-6 image at anchor 3")
        with Image.open(root / "0.png") as first, \
                Image.open(root / "3.png") as third:
            reject(lambda: build_live_inputs(
                processor, record["input"]["instruction"],
                {0: first, 3: third}, history, 3),
                "future actions at anchor 3")
        contaminated = json.loads(json.dumps(record))
        contaminated["input"]["oracle_distance"] = 2.0
        reject(lambda: build_inputs(processor, contaminated, root, 3),
               "privileged distance in model input")
    print(json.dumps({"schema": "future_advantage_visual_parity_v1",
                      "anchors": counts, "offline_live_tensor_equal": True,
                      "future_and_privileged_inputs_rejected": True}))


if __name__ == "__main__":
    main()
