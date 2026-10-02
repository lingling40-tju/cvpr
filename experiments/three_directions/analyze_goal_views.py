"""Audit frozen panorama/text retrieval on natural train-only goal pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def evaluate(data: dict, pairs: list[dict], scenes: set[str]) -> dict:
    ids = data["episode_ids"].tolist()
    index = {eid: i for i, eid in enumerate(ids)}
    image, text = data["images"], data["texts"]
    outcome = []
    for pair in pairs:
        if pair["scene"] not in scenes:
            continue
        a, b = index[pair["left"]], index[pair["right"]]
        if data["scene_ids"][a] != pair["scene"] or data["scene_ids"][b] != pair["scene"]:
            raise ValueError("scene mismatch")
        aa = (image[a] * text[a]).sum(-1).max().item()
        ab = (image[a] * text[b]).sum(-1).max().item()
        ba = (image[b] * text[a]).sum(-1).max().item()
        bb = (image[b] * text[b]).sum(-1).max().item()
        outcome.append({"left": pair["left"], "right": pair["right"],
                        "scene": pair["scene"],
                        "image_text_hits": [aa > ab, bb > ba],
                        "text_image_hits": [aa > ba, bb > ab],
                        "margins": [aa - ab, bb - ba, aa - ba, bb - ab]})
    if not outcome:
        raise ValueError("empty pair subset")
    observed_scenes = {row["scene"] for row in outcome}
    return {"scenes": len(observed_scenes), "pairs": len(outcome),
            "image_text_accuracy": sum(sum(r["image_text_hits"]) for r in outcome) / (2 * len(outcome)),
            "text_image_accuracy": sum(sum(r["text_image_hits"]) for r in outcome) / (2 * len(outcome)),
            "per_scene": [{"scene": scene,
                           "pairs": sum(r["scene"] == scene for r in outcome),
                           "image_text_accuracy": sum(sum(r["image_text_hits"])
                                                      for r in outcome if r["scene"] == scene) /
                               (2 * sum(r["scene"] == scene for r in outcome))}
                          for scene in sorted(observed_scenes)],
            "rows": outcome}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    fit = torch.load(args.fit_features, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_features, map_location="cpu", weights_only=False)
    for subset, data in (("fit", fit), ("calibration", calibration)):
        if data["subset"] != subset or data["episode_ids"].tolist() != \
                manifest["subsets"][subset]["episode_ids"]:
            raise ValueError("manifest coverage mismatch")
    if fit["backbone"] != calibration["backbone"] or \
            fit["model_config_sha256"] != calibration["model_config_sha256"]:
        raise ValueError("backbone mismatch")
    scenes = manifest["calibration_scenes"]
    report = {"interpretation": "Frozen train-scene goal panorama diagnostic; no RL result.",
              "fit": evaluate(fit, manifest["subsets"]["fit"]["pairs"], set(manifest["fit_scenes"])),
              "development": evaluate(calibration, manifest["subsets"]["calibration"]["pairs"],
                                      set(scenes[:5])),
              "audit": evaluate(calibration, manifest["subsets"]["calibration"]["pairs"],
                                set(scenes[5:]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: {name: value for name, value in report[key].items() if name not in ("rows", "per_scene")}
                      for key in ("fit", "development", "audit")}, indent=2))


if __name__ == "__main__":
    main()
