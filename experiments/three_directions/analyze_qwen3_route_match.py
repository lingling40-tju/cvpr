"""Verify four Qwen3-VL route shards and compare with fixed SigLIP baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--siglip-baseline", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "calibration"), default="calibration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ordinal = json.loads(args.ordinal_manifest.read_text())
    expected = ordinal["subsets"][args.part]["pairs"]
    if len(expected) != {"fit": 128, "calibration": 32}[args.part]:
        raise ValueError("natural pair source changed")
    baseline = json.loads(args.siglip_baseline.read_text())
    baseline_rows = {(p["episode_a"], p["episode_b"]): p
                     for p in baseline["partitions"][args.part]["per_pair"]}
    expected_keys = {(p["left"], p["right"]) for p in expected}
    if set(baseline_rows) != expected_keys:
        raise ValueError("baseline pair source mismatch")
    shards = []
    shard_hashes = {}
    for shard in range(4):
        path = args.shard_root / f"{args.part}_shard{shard}.json"
        data = json.loads(path.read_text())
        if data["schema"] != "qwen3_route_match_shard_v1" or data["smoke_only"] or \
                data["shard"] != shard or data["shards"] != 4 or \
                data["pairs"] != {"fit": 32, "calibration": 8}[args.part] or \
                data["routes"] != {"fit": 64, "calibration": 16}[args.part] or \
                data["queries"] != {"fit": 128, "calibration": 32}[args.part] or \
                data["token_ids"] != {"A": 32, "B": 33} or \
                data["max_image_edge"] != 448:
            raise ValueError(f"incomplete shard {shard}")
        if data.get("part", "calibration") != args.part:
            raise ValueError(f"part mismatch shard {shard}")
        if any(row["episode_ids"] != [pair["left"], pair["right"]]
               for row, pair in zip(data["results"], expected[shard::4])):
            raise ValueError(f"pair order mismatch shard {shard}")
        shards.append(data)
        shard_hashes[str(shard)] = digest(path)
    source = shards[0]["source_sha256"]
    prompt_sha = shards[0]["prompt_sha256"]
    if any(s["source_sha256"] != source or s["prompt_sha256"] != prompt_sha
           for s in shards):
        raise ValueError("source/model/prompt hashes differ across shards")
    if source["ordinal_manifest"] != digest(args.ordinal_manifest):
        raise ValueError("ordinal hash mismatch")
    rows = []
    scenes = defaultdict(lambda: {"model": [], "baseline": [], "order_agreement": []})
    for shard in shards:
        for row in shard["results"]:
            key = tuple(row["episode_ids"])
            old = baseline_rows[key]
            if row["scene_id"] != old["scene_id"] or len(row["routes"]) != 2:
                raise ValueError("scene or route mismatch")
            margins = [r["correct_margin_average"] for r in row["routes"]]
            first = [r["correct_margin_first"] for r in row["routes"]]
            swapped = [r["correct_margin_swapped"] for r in row["routes"]]
            baseline_margins = old["rule_margins"]["whole"]
            correct = [int(v > 0) for v in margins]
            baseline_correct = [int(v > 0) for v in baseline_margins]
            consistent = [int(a * b > 0) for a, b in zip(first, swapped)]
            scenes[row["scene_id"]]["model"].extend(correct)
            scenes[row["scene_id"]]["baseline"].extend(baseline_correct)
            scenes[row["scene_id"]]["order_agreement"].extend(consistent)
            rows.append({"episode_ids": row["episode_ids"], "scene_id": row["scene_id"],
                         "model_margins": margins, "baseline_margins": baseline_margins,
                         "first_order_margins": first, "swapped_order_margins": swapped,
                         "correct": correct, "baseline_correct": baseline_correct,
                         "order_agreement": consistent,
                         "input_tokens": [r["input_tokens"] for r in row["routes"]],
                         "prompt_sha256": [r["prompt_sha256"] for r in row["routes"]]})
    if len(rows) != len(expected) or {tuple(row["episode_ids"]) for row in rows} != expected_keys:
        raise ValueError("merged episode pair coverage mismatch")
    model_correct = sum(sum(row["correct"]) for row in rows)
    old_correct = sum(sum(row["baseline_correct"]) for row in rows)
    order_agreement = sum(sum(row["order_agreement"]) for row in rows)
    if old_correct != {"fit": 177, "calibration": 48}[args.part]:
        raise ValueError("frozen full-instruction baseline changed")
    model_scene_macro = sum(sum(v["model"]) / len(v["model"]) for v in scenes.values()) / len(scenes)
    baseline_scene_macro = sum(sum(v["baseline"]) / len(v["baseline"]) for v in scenes.values()) / len(scenes)
    rng = random.Random(11)
    clusters = list(scenes.values())
    differences = []
    for _ in range(2000):
        sample = [rng.choice(clusters) for _ in clusters]
        differences.append(sum(sum(v["model"]) - sum(v["baseline"]) for v in sample) /
                           sum(len(v["model"]) for v in sample))
    differences.sort()
    threshold = {"fit": 190, "calibration": 52}[args.part]
    decisions = 2 * len(expected)
    passed = model_correct >= threshold and model_scene_macro > baseline_scene_macro and \
             order_agreement >= 0.75 * decisions
    report = {"schema": "qwen3_route_match_analysis_v1", "part": args.part,
              "interpretation": "exploratory R2R-train scene route grounding only; no policy reward or navigation result",
              "source_sha256": {**source, "prompt": prompt_sha,
                                "siglip_baseline": digest(args.siglip_baseline),
                                "shards": shard_hashes},
              "coverage": {"natural_pairs": len(expected), "unique_routes": decisions,
                           "model_queries": 2 * decisions, "images_per_query": 6,
                           "scenes": len(scenes)},
              "metrics": {"model_correct": model_correct, "baseline_correct": old_correct,
                          "comparisons": decisions, "model_accuracy": model_correct / decisions,
                          "baseline_accuracy": old_correct / decisions,
                          "model_strict_pairs_correct": sum(all(row["correct"]) for row in rows),
                          "baseline_strict_pairs_correct": sum(all(row["baseline_correct"]) for row in rows),
                          "model_scene_macro_accuracy": model_scene_macro,
                          "baseline_scene_macro_accuracy": baseline_scene_macro,
                          "order_agreement": order_agreement,
                          "order_agreement_rate": order_agreement / decisions,
                          "paired_scene_bootstrap_difference_95":
                              [differences[50], differences[1949]]},
              "gate": {"minimum_correct": threshold,
                       "minimum_order_agreement": int(0.75 * decisions),
                       "scene_macro_must_exceed_baseline": True,
                       "passed": passed}, "per_pair": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coverage": report["coverage"], "metrics": report["metrics"],
                      "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
