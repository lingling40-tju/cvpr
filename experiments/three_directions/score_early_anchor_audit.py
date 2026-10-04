"""Score frozen turn-3 RGB prefixes without reading geodesic audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from train_anchor_potential_lora import digest, load_model, score


MANIFEST_SHA = "dfd9dd4eb663cc05c1c64b4a1d7689b9ad64f72e41a4ac97e6ea990ed66d85a1"
CHECKPOINT_SHA = "0b955f0cde2d77a89f48f0d72e2346c65afaf1b46bd1578d716caf4ae5ef3729"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "rgb-root", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if digest(args.manifest) != MANIFEST_SHA or \
            digest(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("frozen audit manifest or checkpoint changed")
    manifest = json.loads(args.manifest.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if manifest.get("schema") != "early_anchor_group4_audit_source_manifest_v1" or \
            manifest.get("selected_records") != 540 or \
            saved.get("schema") != "anchor_distance_potential_lora_final_v1" or \
            saved.get("microsteps") != 1024 or \
            saved.get("model_config_sha256") != digest(args.model / "config.json") or \
            saved.get("visual_input_sha256") != digest(
                Path(__file__).with_name("anchor_potential_visual_input.py")):
        raise ValueError("audit source or observation input changed")
    processor, model, head = load_model(args.model)
    expected = get_peft_model_state_dict(model)
    if set(saved["adapter"]) != set(expected) or any(
            saved["adapter"][key].shape != expected[key].shape
            for key in expected):
        raise ValueError("checkpoint LoRA keys or shapes changed")
    set_peft_model_state_dict(model, saved["adapter"])
    head.load_state_dict(saved["head"], strict=True)
    model.eval()
    head.eval()
    rgb_part = args.rgb_root / "audit"
    lines = []
    with torch.inference_mode():
        for index, plan in enumerate(manifest["plans"], 1):
            rid = plan["record_id"]
            record = json.loads((rgb_part / "records" / f"{rid}.json").read_text())
            if record.get("record_id") != rid or \
                    record.get("manifest_sha256") != MANIFEST_SHA or \
                    set(record["input"]) != {"instruction", "images",
                                             "action_history_by_anchor"}:
                raise ValueError(f"invalid observation-only record: {rid}")
            value = float(score(processor, model, head, record, rgb_part, 3))
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"nonfinite potential score: {rid}")
            lines.append(json.dumps({"record_id": rid, "score": value}))
            if index % 50 == 0:
                print(f"scored {index}/{len(manifest['plans'])}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(args.output)
    summary = {
        "schema": "early_anchor_observation_only_scores_v1",
        "manifest_sha256": MANIFEST_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "records": len(lines),
        "scores_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "interpretation": "Frozen model scores only; no geodesic labels or ranking result",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
