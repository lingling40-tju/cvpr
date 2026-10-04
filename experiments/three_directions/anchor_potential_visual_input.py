"""Observation-only prefix prompt for current instruction-conditioned proximity."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from future_advantage_visual_input import _actions, _path, _prepared


PREFIX = (
    "Assess the agent's current proximity to the instructed destination "
    "from the navigation instruction, visible views, and executed actions. "
    "Use only evidence available through the current view."
)
SUFFIX = "Current proximity to the instructed destination:"


def build_inputs(processor, record: dict, root: Path, anchor: int):
    if record.get("schema") != "future_advantage_sparse_model_input_v1" or \
            anchor not in (3, 6):
        raise ValueError("invalid sparse prefix or anchor")
    fields = record["input"]
    if set(fields) != {"instruction", "images", "action_history_by_anchor"}:
        raise ValueError("privileged or missing model input")
    instruction = fields["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("missing navigation instruction")
    needed = ["0", "3"] if anchor == 3 else ["0", "3", "6"]
    if any(key not in fields["images"] for key in needed):
        raise ValueError("missing prefix RGB")
    history = fields["action_history_by_anchor"].get(str(anchor))
    if history is None or len(history) != anchor or \
            [row["turn"] for row in history] != list(range(1, anchor + 1)):
        raise ValueError("missing or future action history")
    frames = []
    try:
        for key in needed:
            with Image.open(_path(root, fields["images"][key])) as source:
                frames.append(_prepared(source.convert("RGB")))
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
            raise ValueError("anchor potential prefix exceeds 12000 tokens")
        return inputs
    finally:
        for frame in frames:
            frame.close()
