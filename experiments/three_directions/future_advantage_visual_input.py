"""Build a label-free multimodal prefix input from sparse n=4 RGB records."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from qwen_vl_utils import fetch_image


MAX_PIXELS = 76800
PREFIX = (
    "Assess this navigation route prefix from the instruction, views, and "
    "executed actions. Use only evidence available through the current view."
)
SUFFIX = "Expected future progress toward the instructed destination:"


def _path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("image path must stay inside the replay root")
    full = root / path
    if not full.is_file():
        raise ValueError(f"missing sparse RGB image: {relative}")
    return full


def _actions(rows: list[dict], start: int, end: int) -> str:
    picked = [row for row in rows if start <= row["turn"] <= end]
    if [row["turn"] for row in picked] != list(range(start, end + 1)):
        raise ValueError("incomplete or reordered executed turn history")
    chunks = []
    for row in picked:
        actions = row["executed_actions"]
        if not actions or any("stop" in action.lower() for action in actions):
            raise ValueError("STOP or empty action in active prefix")
        chunks.append(f"turn {row['turn']}: " + ", ".join(actions))
    return " | ".join(chunks)


def _prepared(raw: Image.Image) -> Image.Image:
    processed = fetch_image({"image": raw, "max_pixels": MAX_PIXELS,
                             "min_pixels": 1024})
    if processed is not raw:
        raw.close()
    return processed


def _with_frames(processor, instruction: str, frames: list[Image.Image],
                 history: list[dict], anchor: int):
    if anchor not in (3, 6) or not instruction.strip() or \
            len(frames) != (2 if anchor == 3 else 3):
        raise ValueError("invalid sparse route prefix")
    if len(history) != anchor or [row["turn"] for row in history] != \
            list(range(1, anchor + 1)):
        raise ValueError("history includes missing or future turns")
    content = [
        {"type": "text", "text": f"{PREFIX}\nNavigation instruction: {instruction.strip()}\nInitial view:"},
        {"type": "image", "image": frames[0]},
        {"type": "text", "text": "\nExecuted " + _actions(history, 1, 3) +
         "\nView after turn 3:"},
        {"type": "image", "image": frames[1]},
    ]
    if anchor == 6:
        content.extend([
            {"type": "text", "text": "\nExecuted " +
             _actions(history, 4, 6) + "\nView after turn 6:"},
            {"type": "image", "image": frames[2]},
        ])
    content.append({"type": "text", "text": "\n" + SUFFIX})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    if int(inputs["input_ids"].shape[1]) > 12000:
        raise ValueError("sparse prefix exceeds 12000 tokens")
    return inputs


def build_inputs(processor, record: dict, root: Path, anchor: int):
    if record.get("schema") != "future_advantage_sparse_model_input_v1" or \
            anchor not in (3, 6):
        raise ValueError("invalid sparse prefix record or anchor")
    fields = record["input"]
    if set(fields) != {"instruction", "images", "action_history_by_anchor"}:
        raise ValueError("privileged or missing field in model input")
    images = fields["images"]
    history = fields["action_history_by_anchor"].get(str(anchor))
    needed = ["0", "3"] if anchor == 3 else ["0", "3", "6"]
    if history is None or any(index not in images for index in needed):
        raise ValueError("incomplete sparse route prefix")
    frames = []
    try:
        for index in needed:
            with Image.open(_path(root, images[index])) as source:
                raw = source.convert("RGB")
            frames.append(_prepared(raw))
        return _with_frames(processor, fields["instruction"], frames,
                            history, anchor)
    finally:
        for frame in frames:
            frame.close()


def build_live_inputs(processor, instruction: str,
                      frames_by_turn: dict[int, Image.Image],
                      executed_history: list[dict], anchor: int):
    """Apply the exact offline prompt to in-memory live Habitat RGB views."""
    needed = [0, 3] if anchor == 3 else [0, 3, 6] if anchor == 6 else []
    if set(frames_by_turn) != set(needed) or any(
            not isinstance(frame, Image.Image)
            for frame in frames_by_turn.values()):
        raise ValueError("live frames must contain only available prefix views")
    frames = []
    try:
        for index in needed:
            frames.append(_prepared(frames_by_turn[index].convert("RGB")))
        return _with_frames(processor, instruction, frames,
                            executed_history, anchor)
    finally:
        for frame in frames:
            frame.close()
