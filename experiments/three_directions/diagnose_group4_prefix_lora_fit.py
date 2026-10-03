"""Small descriptive fit-scene instruction diagnostic; never a model gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model
from fit_group4_future_success_linear import ANCHORS


def key(item: tuple[dict, int]) -> str:
    row, anchor = item
    return hashlib.sha256(
        f"group4-prefix-fit-diagnostic-v1:{row['episode_id']}:{anchor}".encode()
    ).hexdigest()


def margin(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor,
           vector: torch.Tensor) -> float:
    return float(F.normalize((a.float() - b.float()) / scale, dim=0)
                 @ vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--old-state-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    manifest_sha = digest(args.manifest)
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1" or \
            checkpoint["schema"] != "group4_prefix_contrast_lora_v1" or \
            checkpoint["source_sha256"]["expert_manifest"] != manifest_sha or \
            checkpoint["source_sha256"]["frozen_weights"] != digest(args.frozen_weights) or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
            weights["schema"] != "group4_joint_value_weights_v1":
        raise ValueError("fit diagnostic source mismatch")
    by_scene = defaultdict(list)
    for row in manifest["selected"]["fit"]:
        for anchor in ANCHORS:
            if anchor < row["turn_count"]:
                by_scene[row["scene_id"]].append((row, anchor))
    if len(by_scene) != 38 or sum(map(len, by_scene.values())) != 1107:
        raise ValueError("fit diagnostic population changed")
    chosen = [item for scene in sorted(by_scene)
              for item in sorted(by_scene[scene], key=key)[:2]]
    if len(chosen) != 76:
        raise ValueError("fit diagnostic sample changed")
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    scale, vector = weights["scale"].float(), weights["vector"].float()
    records = []
    with torch.inference_mode():
        for number, (row, anchor) in enumerate(chosen, 1):
            eid = str(row["episode_id"])
            record_path = args.record_root / row["source_part"] / "records" / f"{eid}.json"
            old_path = args.old_state_root / "fit" / "records" / f"{eid}.pt"
            if digest(record_path) != row["record_sha256"]:
                raise ValueError(f"fit record changed {eid}")
            record = json.loads(record_path.read_text())
            old = torch.load(old_path, map_location="cpu", weights_only=True)
            if old["schema"] != "group4_expert_prefix_state_v1" or \
                    old["manifest_sha256"] != manifest_sha or \
                    old["source_id"] != weights["encoder_source_id"] or \
                    old["record_sha256"] != row["record_sha256"] or \
                    anchor not in old["anchors"]:
                raise ValueError(f"old fit cache changed {eid}/{anchor}")
            index = old["anchors"].index(anchor)
            old_margin = margin(old["correct"][index], old["wrong"][index],
                                scale, vector)
            item = {"record": record,
                    "root": args.record_root / row["source_part"]}
            values = []
            for instruction in (record["instruction"], record["wrong_instruction"]):
                inputs = build_inputs(processor, item, anchor,
                                      instruction).to("cuda")
                hidden = model(**inputs, output_hidden_states=False,
                               use_cache=False).logits[0, -1].float().cpu()
                if hidden.shape != (2048,) or \
                        not bool(torch.isfinite(hidden).all()):
                    raise ValueError(f"nonfinite fit state {eid}/{anchor}")
                values.append(hidden.to(torch.float16))
            new_margin = margin(values[0], values[1], scale, vector)
            records.append({"episode_id": eid, "anchor": anchor,
                            "scene_id": row["scene_id"],
                            "old_margin": old_margin,
                            "new_margin": new_margin})
            if number % 20 == 0:
                print(f"fit diagnostic {number}/{len(chosen)}", flush=True)
    def summarize(name: str) -> dict:
        correct = sum(row[f"{name}_margin"] > 0 for row in records)
        macro = sum(sum(row[f"{name}_margin"] > 0 for row in records
                        if row["scene_id"] == scene) / 2 for scene in by_scene) / 38
        return {"correct": correct, "pairs": len(records),
                "accuracy": correct / len(records),
                "scene_macro_accuracy": macro,
                "mean_margin": sum(row[f"{name}_margin"] for row in records)
                               / len(records)}
    result = {"schema": "group4_prefix_lora_fit_diagnostic_v1",
              "interpretation": "fixed small fit-scene sample, descriptive only; includes training examples and cannot gate a model or support navigation claims",
              "source_sha256": {"manifest": manifest_sha,
                                "checkpoint": digest(args.checkpoint),
                                "frozen_weights": digest(args.frozen_weights)},
              "selection": "SHA-top-two nonterminal prefix contrasts per fit scene",
              "scenes": len(by_scene), "pairs": len(records),
              "old": summarize("old"), "adapted": summarize("new")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
