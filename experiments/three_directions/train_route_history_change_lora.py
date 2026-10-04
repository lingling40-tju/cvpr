"""Route-history input variant of the frozen categorical change probe.

The target is the same R2R-train geodesic turn label. At inference the model
sees only the instruction, executed action text, and RGB observations at
selected preceding turn boundaries. The base probe owns the unchanged fit,
development split, losses, optimizer, checkpoint schedule, and gates.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import train_balanced_change_lora as core
from history_grounding_lora import digest
from train_action_memory_progress_lora import action_text
from train_joint_pair_progress_lora import frame_path, image


def route_boundary_indices(before: int, after: int) -> list[int]:
    """At most four prefix views, chosen without future observations."""
    if not 0 <= before < after:
        raise ValueError("invalid route turn span")
    last_prefix = before if after == before + 1 else after - 1
    return sorted({0, before, last_prefix // 3,
                   (2 * last_prefix) // 3, last_prefix})


def build_history_inputs(processor, item: dict, before: int, after: int,
                         instruction: str | None = None):
    record = item["record"]
    if after > len(record["turns"]):
        raise ValueError("route turn index exceeds record")
    boundaries = route_boundary_indices(before, after)
    views = [image(frame_path(item, index))
             for index in boundaries + [after]]
    try:
        parts = ["Navigation instruction: " +
                 (record["instruction"] if instruction is None else
                  instruction).strip(),
                 "Observed route history in chronological order:"]
        for position, index in enumerate(boundaries):
            if position:
                previous = boundaries[position - 1]
                parts.append("Executed actions between these views: " +
                             action_text(item, previous, index))
            parts.append(f"Route boundary {index}: <image>")
        parts.extend(("Executed action sequence being judged: " +
                      action_text(item, before, after),
                      f"View after the judged action sequence at boundary {after}: <image>",
                      "Classify visual progress toward the instruction's goal "
                      "as forward, backward, or stationary."))
        message = {"role": "user", "content": "\n".join(parts)}
        prompt = processor.tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True)
        prompt = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "",
                        prompt, flags=re.S)
        if prompt.count("<image>") != len(views):
            raise ValueError("route-history image count changed")
        prompt = prompt.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = processor(text=[prompt], images=views, return_tensors="pt")
        if int(inputs["input_ids"].shape[1]) > 16000:
            raise ValueError("route-history prompt exceeds 16000 tokens")
        return inputs
    finally:
        for view in views:
            view.close()


def main() -> None:
    core.build_inputs = build_history_inputs
    core.main()
    if "--smoke" in sys.argv:
        return
    index = sys.argv.index("--output")
    output = Path(sys.argv[index + 1])
    report_path = output / "development.json"
    report = json.loads(report_path.read_text())
    if report["schema"] != "balanced_visual_change_lora_development_v1":
        raise ValueError("base report schema changed")
    report["schema"] = "route_history_change_lora_development_v1"
    report["input_variant"] = "up_to_four_preceding_rgb_boundaries_plus_after"
    report["source_sha256"]["history_prompt"] = digest(Path(__file__))
    report["source_sha256"]["balanced_change_base"] = digest(
        Path(core.__file__))
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
