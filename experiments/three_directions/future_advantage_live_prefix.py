"""Encode live observations exactly like the sparse Habitat replay collector.

This module is deliberately CPU-only. The rollout worker can form one n=4
request without importing the Qwen model or loading its reward checkpoint.
"""

from __future__ import annotations

import base64
import io

from PIL import Image


ANCHOR_IMAGES = {3: (0, 3), 6: (0, 3, 6)}


def canonical_jpeg(frame) -> str:
    """Match collect_policy_preference_frames.save_frame's RGB/JPEG path."""
    if isinstance(frame, Image.Image):
        image = frame.convert("RGB")
    else:
        image = Image.fromarray(frame).convert("RGB")
    try:
        image.thumbnail((336, 336))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        image.close()


def score_item(instruction: str, frames_by_turn: dict,
               executed_history: list[dict], anchor: int) -> dict:
    needed = ANCHOR_IMAGES.get(anchor)
    if needed is None or not isinstance(instruction, str) or \
            not instruction.strip() or len(instruction) > 4096 or \
            set(frames_by_turn) != set(needed) or \
            len(executed_history) != anchor or \
            any(not isinstance(row, dict) or set(row) != {
                "turn", "executed_actions"} for row in executed_history) or \
            [row.get("turn") for row in executed_history] != \
            list(range(1, anchor + 1)):
        raise ValueError("live prefix has missing or future observation/action")
    if any(not isinstance(row["executed_actions"], list) or
           not 1 <= len(row["executed_actions"]) <= 16 or
           any(not isinstance(action, str) or
               not 0 < len(action) <= 80 or "stop" in action.lower()
               for action in row["executed_actions"])
           for row in executed_history):
        raise ValueError("live prefix must contain executed movement only")
    return {"instruction": instruction, "anchor": anchor,
            "images": {str(turn): canonical_jpeg(frames_by_turn[turn])
                       for turn in needed},
            "history": [{"turn": row["turn"],
                         "executed_actions": list(row["executed_actions"])}
                        for row in executed_history]}
