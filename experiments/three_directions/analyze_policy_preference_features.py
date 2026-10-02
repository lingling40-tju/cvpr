"""Frozen-feature success/failure ranking on train-scene policy pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.features, map_location="cpu", weights_only=False)
    if data["schema"] != "policy_preference_features_v1" or \
            data["images"].shape[0] != 800 or data["images"].shape[1] != 4 or \
            len(data["record_ids"]) != 800 or len(data["scene_ids"]) != 800:
        raise ValueError("incomplete policy feature cache")
    if any(data["roles"][index:index + 2] != ["success", "failure"] or
           data["pair_ids"][index] != data["pair_ids"][index + 1] or
           data["splits"][index] != data["splits"][index + 1] or
           data["scene_ids"][index] != data["scene_ids"][index + 1]
           for index in range(0, 800, 2)):
        raise ValueError("pair order mismatch")
    similarity = (data["images"] * data["texts"][:, None, :]).sum(-1)
    results = defaultdict(list)
    for index in range(0, 800, 2):
        success, failure = similarity[index:index + 2]
        result = {"pair_id": data["pair_ids"][index],
                  "scene": data["scene_ids"][index],
                  "last_frame_hit": bool(success[-1] > failure[-1]),
                  "last_two_max_hit": bool(success[-2:].max() > failure[-2:].max()),
                  "start_relative_hit": bool(success[-1] - success[0] >
                                              failure[-1] - failure[0]),
                  "last_two_max_margin": float(success[-2:].max() - failure[-2:].max())}
        results[data["splits"][index]].append(result)
    report = {"interpretation": "Frozen SigLIP train-scene policy outcome screen; no RL result.",
              "manifest_sha256": data["manifest_sha256"]}
    for split in ("fit", "development", "audit"):
        rows = results[split]
        if not rows:
            raise ValueError(f"empty {split}")
        scenes = sorted({row["scene"] for row in rows})
        report[split] = {"pairs": len(rows), "scenes": len(scenes),
                         "last_frame_accuracy": sum(row["last_frame_hit"] for row in rows) / len(rows),
                         "last_two_max_accuracy": sum(row["last_two_max_hit"] for row in rows) / len(rows),
                         "start_relative_accuracy": sum(row["start_relative_hit"] for row in rows) / len(rows),
                         "last_two_max_margin_mean": sum(row["last_two_max_margin"] for row in rows) / len(rows),
                         "per_scene": [{"scene": scene,
                                        "pairs": sum(row["scene"] == scene for row in rows),
                                        "last_two_max_accuracy":
                                            sum(row["last_two_max_hit"] for row in rows
                                                if row["scene"] == scene) /
                                            sum(row["scene"] == scene for row in rows)}
                                       for scene in scenes],
                         "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {field: value for field, value in report[key].items()
                            if field not in ("rows", "per_scene")}
                      for key in ("fit", "development", "audit")}, indent=2))


if __name__ == "__main__":
    main()
