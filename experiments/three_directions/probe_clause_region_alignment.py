"""Fixed zero-shot local-region versus whole-instruction grounding screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch

from prepare_clause_alignment_manifest import digest


RULES = ("whole_instruction", "region_ordered_path_gain",
         "full_frame_ordered_path_gain", "region_final_clause_gain")


def path_score(similarity: torch.Tensor) -> float:
    length = similarity.shape[1]
    dp = torch.full((length,), -float("inf"), dtype=torch.float32)
    dp[0] = similarity[0, 0]
    for t in range(1, 6):
        dp = similarity[t] + torch.cummax(dp, dim=0).values
    return float(dp[-1] / 6)


def scores(regions: torch.Tensor, image_full: torch.Tensor,
           instruction_full: torch.Tensor, clauses: torch.Tensor) -> dict:
    similarity = torch.einsum("trd,kd->trk", regions, clauses)
    best = similarity.max(dim=1).values
    full = similarity[:, 0, :]
    return {
        "whole_instruction": float(image_full[-2:].mean(dim=0) @ instruction_full),
        "region_ordered_path_gain": path_score(best) - path_score(best[:1].expand_as(best)),
        "full_frame_ordered_path_gain": path_score(full) - path_score(full[:1].expand_as(full)),
        "region_final_clause_gain": float(best[-2:, -1].max() - best[0, -1]),
    }


def load_part(part: str, args: argparse.Namespace, manifest: dict,
              ordinal: dict, audit: dict) -> dict:
    rows = manifest["selected"][part]
    text_path = args.clause_root / f"{part}_text.pt"
    text = torch.load(text_path, map_location="cpu", weights_only=True)
    visual_path = args.ordinal_root / f"siglip_full_{part}.pt"
    visual = torch.load(visual_path, map_location="cpu", weights_only=True)
    if text["episode_ids"].tolist() != [row["episode_id"] for row in rows] or \
            visual["episode_ids"].tolist() != [row["episode_id"] for row in rows] or \
            text["model_config_sha256"] != audit["source_hashes"]["config_sha256"] or \
            visual["model_config_sha256"] != text["model_config_sha256"] or \
            text["manifest_sha256"] != digest(args.manifest):
        raise ValueError(f"source mismatch: {part}")
    by_episode = defaultdict(list)
    for j, (eid, offset) in enumerate(zip(visual["frame_episode_ids"].tolist(),
                                          visual["frame_offsets"].tolist())):
        by_episode[eid].append((offset, j))
    offsets = text["offsets"].tolist()
    images, regions, clauses = [], [], []
    for row, start, end in zip(rows, offsets[:-1], offsets[1:]):
        eid = row["episode_id"]
        indices = [j for _, j in sorted(by_episode[eid])]
        if len(indices) != 6 or [visual["frame_offsets"][j].item()
                                 for j in indices] != list(range(6)) or \
                end - start != row["clause_count"]:
            raise ValueError(f"frame/clause coverage mismatch: {part}/{eid}")
        payload = torch.load(args.region_root / part / "records" / f"{eid}.pt",
                             map_location="cpu", weights_only=True)
        if payload["episode_id"] != eid or payload["record_sha256"] != row["record_sha256"] or \
                payload["model_sha256"] != audit["source_hashes"]["model_sha256"]:
            raise ValueError(f"region identity mismatch: {part}/{eid}")
        images.append(visual["images"][indices].float())
        regions.append(payload["features"].float())
        clauses.append(text["clauses"][start:end].float())
    pairs = ordinal["subsets"][part]["pairs"]
    if len(pairs) != {"fit": 128, "calibration": 32}[part]:
        raise ValueError("natural pair count changed")
    return {"rows": rows, "by_id": {row["episode_id"]: i for i, row in enumerate(rows)},
            "images": images, "regions": regions, "clauses": clauses,
            "full": text["full"].float(), "pairs": pairs,
            "source_hashes": {"text": digest(text_path), "visual": digest(visual_path)}}


def summarize(part: dict) -> dict:
    by_scene = defaultdict(lambda: {rule: [] for rule in RULES})
    per_pair = []
    for pair in part["pairs"]:
        a, b = part["by_id"][pair["left"]], part["by_id"][pair["right"]]
        if part["rows"][a]["scene_id"] != pair["scene"] or \
                part["rows"][b]["scene_id"] != pair["scene"]:
            raise ValueError("pair scene mismatch")
        aa = scores(part["regions"][a], part["images"][a],
                    part["full"][a], part["clauses"][a])
        ab = scores(part["regions"][a], part["images"][a],
                    part["full"][b], part["clauses"][b])
        bb = scores(part["regions"][b], part["images"][b],
                    part["full"][b], part["clauses"][b])
        ba = scores(part["regions"][b], part["images"][b],
                    part["full"][a], part["clauses"][a])
        margins = {rule: [aa[rule] - ab[rule], bb[rule] - ba[rule]] for rule in RULES}
        for rule in RULES:
            by_scene[pair["scene"]][rule].extend(int(v > 0) for v in margins[rule])
        per_pair.append({"episode_ids": [pair["left"], pair["right"]],
                         "scene_id": pair["scene"], "rule_margins": margins})
    metrics = {}
    for rule in RULES:
        values = [v for scene in by_scene.values() for v in scene[rule]]
        metrics[rule] = {"correct": sum(values), "comparisons": len(values),
                         "accuracy": sum(values) / len(values),
                         "strict_pairs_correct": sum(all(v > 0 for v in row["rule_margins"][rule])
                                                     for row in per_pair),
                         "pairs": len(per_pair), "scenes": len(by_scene),
                         "scene_macro_accuracy": sum(sum(scene[rule]) / len(scene[rule])
                                                     for scene in by_scene.values()) / len(by_scene)}
    scenes = list(by_scene.values())
    rng = random.Random(11)
    differences = []
    for _ in range(2000):
        sampled = [rng.choice(scenes) for _ in scenes]
        differences.append(sum(sum(s["region_ordered_path_gain"]) -
                               sum(s["whole_instruction"]) for s in sampled) /
                           sum(len(s["whole_instruction"]) for s in sampled))
    differences.sort()
    return {"metrics": metrics, "paired_primary_minus_baseline_scene_bootstrap_95":
            [differences[50], differences[1949]], "per_pair": per_pair,
            "source_hashes": part["source_hashes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--ordinal-root", type=Path, required=True)
    parser.add_argument("--clause-root", type=Path, required=True)
    parser.add_argument("--region-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    ordinal = json.loads(args.ordinal_manifest.read_text())
    audit = json.loads(args.cache_audit.read_text())
    if manifest["ordinal_manifest_sha256"] != digest(args.ordinal_manifest) or \
            audit["schema"] != "clause_region_cache_audit_v1" or \
            audit["source_hashes"]["manifest_sha256"] != digest(args.manifest) or \
            not audit["scene_disjoint"]:
        raise ValueError("manifest or cache audit mismatch")
    output = {}
    for part in ("fit", "calibration"):
        output[part] = summarize(load_part(part, args, manifest, ordinal, audit))
    candidate = output["calibration"]["metrics"]["region_ordered_path_gain"]
    baseline = output["calibration"]["metrics"]["whole_instruction"]
    if baseline["correct"] != 48:
        raise ValueError("fixed-padding whole-instruction baseline changed")
    passed = candidate["correct"] >= 52 and \
             candidate["scene_macro_accuracy"] > baseline["scene_macro_accuracy"]
    report = {"schema": "clause_region_alignment_probe_v1",
              "interpretation": "exploratory train-scene frozen representation, no reward or navigation result",
              "primary_rule": "region_ordered_path_gain",
              "diagnostic_rules": ["full_frame_ordered_path_gain", "region_final_clause_gain"],
              "calibration_gate": "primary at least 52/64 and scene macro above frozen whole instruction",
              "source_hashes": {"manifest": digest(args.manifest),
                                "ordinal_manifest": digest(args.ordinal_manifest),
                                "cache_audit": digest(args.cache_audit)},
              "partitions": output, "calibration_gate_passed": passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fit": output["fit"]["metrics"],
                      "calibration": output["calibration"]["metrics"],
                      "calibration_gate_passed": passed}, indent=2))


if __name__ == "__main__":
    main()
