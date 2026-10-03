"""Freeze a new scene split for a cross-modal backbone adaptation screen.

The new audit scenes come from the earlier readout's fit scenes and were
never used to select that readout's checkpoint or STOP threshold. They
were nevertheless included in earlier representation training, so this
is a new train-scene screen, not untouched external validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SALT = "stop-lora-crossmodal-v1:"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.label_audit.read_text())
    if source["schema"] != "stop_history_train_only_label_audit_v1":
        raise ValueError("wrong source schema")
    old_fit = sorted({row["scene_id"] for row in source["labels"]["fit"]},
                     key=lambda scene: hashlib.sha256((SALT + scene).encode()).hexdigest())
    if len(old_fit) != 36:
        raise ValueError("unexpected previous fit scene coverage")
    new_dev, new_audit = set(old_fit[:8]), set(old_fit[8:16])
    old_dev_audit = {row["scene_id"] for part in ("development", "audit")
                     for row in source["labels"][part]}
    new_fit = set(old_fit[16:]) | old_dev_audit
    split = {"fit": sorted(new_fit), "development": sorted(new_dev),
             "audit": sorted(new_audit)}
    if sum(map(len, split.values())) != len(set().union(*map(set, split.values()))):
        raise ValueError("new scene leakage")
    rows = [row for part in source["labels"].values() for row in part]
    selected = {}
    counts = {}
    for part, scenes in split.items():
        members = [row for row in rows if row["scene_id"] in set(scenes) and
                   row["within_12_turns"]]
        selected[part] = sorted([{"episode_id": str(row["episode_id"]),
                                  "scene_id": row["scene_id"],
                                  "trajectory_id": row["trajectory_id"],
                                  "safe_wrong_instruction": row["safe_wrong_instruction"]}
                                 for row in members], key=lambda row: row["episode_id"])
        counts[part] = {"scenes": len(scenes), "histories": len(members),
                        "safe_wrong_instruction": sum(row["safe_wrong_instruction"]
                                                      for row in members)}
    if counts["development"]["safe_wrong_instruction"] < 100 or \
            counts["audit"]["safe_wrong_instruction"] < 100:
        raise ValueError("new audit/development underpowered")
    all_keys = [(row["scene_id"], row["trajectory_id"])
                for members in selected.values() for row in members]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("trajectory leakage")
    result = {
        "schema": "stop_history_lora_scene_split_v1",
        "label_audit_sha256": digest(args.label_audit),
        "scene_order_rule": "SHA-256(stop-lora-crossmodal-v1: + scene) on earlier fit scenes; first 8 development, next 8 audit; remaining plus earlier development/audit are fit",
        "interpretation": "New scene holdout for a fresh LoRA adaptation, but earlier architecture work used these R2R train scenes. Navigation proof still requires matched val-unseen RL evaluation.",
        "scene_split": split, "counts": counts, "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
