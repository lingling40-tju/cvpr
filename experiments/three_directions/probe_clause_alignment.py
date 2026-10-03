"""Fixed zero-shot clause alignment probe on train-only expert paths.

This probe compares four rules selected before looking at their results.
It does not fit parameters, open VLN val-unseen, or train a policy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch

from prepare_clause_alignment_manifest import digest


RULES = ("whole", "final_clause", "last_two_clauses", "ordered_path",
         "ordered_path_gain")


def ordered_path(images: torch.Tensor, clauses: torch.Tensor) -> float:
    similarity = images @ clauses.T
    length = clauses.shape[0]
    dp = torch.full((length,), -float("inf"), dtype=torch.float32)
    dp[0] = similarity[0, 0]
    for t in range(1, len(images)):
        # Nondecreasing clause index; a jump may cover multiple short clauses
        # between two sampled frames. The first and last clauses are required.
        dp = similarity[t] + torch.cummax(dp, dim=0).values
    return float(dp[-1] / len(images))


def score(images: torch.Tensor, full: torch.Tensor,
          clauses: torch.Tensor) -> dict[str, float]:
    endpoint = images[-2:].mean(dim=0)
    last_two = clauses[-2:].mean(dim=0)
    path = ordered_path(images, clauses)
    initial_repeated = images[:1].expand_as(images)
    return {
        "whole": float(endpoint @ full),
        "final_clause": float(endpoint @ clauses[-1]),
        "last_two_clauses": float(endpoint @ last_two),
        "ordered_path": path,
        "ordered_path_gain": path - ordered_path(initial_repeated, clauses),
    }


def load(part: str, manifest: dict, text_path: Path,
         visual_path: Path) -> tuple[list[dict], torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    text = torch.load(text_path, map_location="cpu", weights_only=True)
    visual = torch.load(visual_path, map_location="cpu", weights_only=True)
    rows = manifest["selected"][part]
    ids = [r["episode_id"] for r in rows]
    offsets = text["offsets"].tolist()
    if text["schema"] != "clause_siglip_text_features_v1" or \
            text["part"] != part or \
            text["manifest_sha256"] != manifest["_sha256"] or \
            text["text_padding_length"] != 64 or \
            text["episode_ids"].tolist() != ids or \
            visual["episode_ids"].tolist() != ids or \
            visual["subset"] != part or \
            text["model_config_sha256"] != visual["model_config_sha256"] or \
            len(offsets) != len(rows) + 1 or \
            offsets[0] != 0 or offsets[-1] != len(text["clauses"]):
        raise ValueError(f"cache mismatch: {part}")
    by_episode = defaultdict(list)
    for j, eid in enumerate(visual["frame_episode_ids"].tolist()):
        by_episode[eid].append(j)
    image_rows = []
    for row_number, (row, start, end) in enumerate(
            zip(rows, offsets[:-1], offsets[1:])):
        eid = row["episode_id"]
        indices = sorted(by_episode[eid],
                         key=lambda j: visual["frame_offsets"][j].item())
        if len(indices) != 6 or \
                [visual["frame_offsets"][j].item() for j in indices] != list(range(6)) or \
                end - start != row["clause_count"]:
            raise ValueError(f"frame or clause coverage mismatch {part}/{eid}")
        if visual["scene_ids"][row_number] != row["scene_id"]:
            raise ValueError(f"scene changed {part}/{eid}")
        image_rows.append(visual["images"][indices].float())
    images = torch.stack(image_rows)
    full = text["full"].float()
    clauses = [text["clauses"][a:b].float() for a, b in zip(offsets[:-1], offsets[1:])]
    if not bool(torch.isfinite(images).all() and torch.isfinite(full).all() and
                torch.isfinite(text["clauses"]).all()):
        raise ValueError("nonfinite clause features")
    return rows, images, full, clauses


def summarize(part: str, rows: list[dict], images: torch.Tensor,
              full: torch.Tensor, clauses: list[torch.Tensor],
              ordinal: dict) -> dict:
    ids = {r["episode_id"]: i for i, r in enumerate(rows)}
    correct = defaultdict(list)
    by_scene = defaultdict(lambda: defaultdict(list))
    per_pair = []
    for pair in ordinal["subsets"][part]["pairs"]:
        ia, ib = ids[pair["left"]], ids[pair["right"]]
        if rows[ia]["scene_id"] != pair["scene"] or \
                rows[ib]["scene_id"] != pair["scene"]:
            raise ValueError(f"natural pair scene changed {part}")
        aa = score(images[ia], full[ia], clauses[ia])
        ab = score(images[ia], full[ib], clauses[ib])
        bb = score(images[ib], full[ib], clauses[ib])
        ba = score(images[ib], full[ia], clauses[ia])
        row = {"episode_a": pair["left"], "episode_b": pair["right"],
               "scene_id": pair["scene"], "clause_counts": [len(clauses[ia]),
                                                                len(clauses[ib])],
               "rule_margins": {}}
        for rule in RULES:
            margin_a, margin_b = aa[rule] - ab[rule], bb[rule] - ba[rule]
            row["rule_margins"][rule] = [margin_a, margin_b]
            correct[rule].extend((int(margin_a > 0), int(margin_b > 0)))
            by_scene[rule][pair["scene"]].extend((int(margin_a > 0),
                                                   int(margin_b > 0)))
        per_pair.append(row)
    if len(per_pair) != ordinal["subsets"][part]["pairs_count"]:
        raise ValueError(f"natural pair coverage mismatch: {part}")
    metrics = {}
    for rule in RULES:
        values = correct[rule]
        scenes = by_scene[rule]
        rng = random.Random(11)
        clusters = list(scenes.values())
        boots = []
        for _ in range(2000):
            sample = [rng.choice(clusters) for _ in clusters]
            boots.append(sum(map(sum, sample)) / sum(map(len, sample)))
        boots.sort()
        metrics[rule] = {
            "comparisons": len(values), "correct": sum(values),
            "accuracy": sum(values) / len(values),
            "scenes": len(scenes),
            "scene_macro_accuracy": sum(sum(v) / len(v) for v in scenes.values()) /
                                    len(scenes),
            "scene_bootstrap_95": [boots[50], boots[1949]],
        }
    return {"pairs": len(per_pair), "metrics": metrics, "per_pair": per_pair}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-manifest", type=Path, required=True)
    parser.add_argument("--fit-text", type=Path, required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--fit-visual", type=Path, required=True)
    parser.add_argument("--calibration-visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest["_sha256"] = digest(args.manifest)
    ordinal = json.loads(args.ordinal_manifest.read_text())
    if manifest["schema"] != "clause_alignment_manifest_v1" or \
            manifest["ordinal_manifest_sha256"] != digest(args.ordinal_manifest):
        raise ValueError("manifest provenance mismatch")
    output = {}
    for part, text_path, visual_path in (
            ("fit", args.fit_text, args.fit_visual),
            ("calibration", args.calibration_text, args.calibration_visual)):
        rows, images, full, clauses = load(part, manifest, text_path, visual_path)
        output[part] = summarize(part, rows, images, full, clauses, ordinal)
    report = {
        "schema": "clause_alignment_zero_shot_probe_v1",
        "interpretation": "train-scene exploratory zero-shot alignment; no policy or navigation result",
        "rules": {"whole": "last-two-frame mean cosine with full fixed-64-token instruction",
                  "final_clause": "last-two-frame mean cosine with final clause",
                  "last_two_clauses": "last-two-frame mean cosine with mean final-two clauses",
                  "ordered_path": "max-sum monotone frame-to-clause path from first to last clause, divided by six",
                  "ordered_path_gain": "ordered path score minus repeated-initial-frame path score"},
        "source_sha256": {
            "manifest": manifest["_sha256"],
            "ordinal_manifest": digest(args.ordinal_manifest),
            "fit_text": digest(args.fit_text),
            "calibration_text": digest(args.calibration_text),
            "fit_visual": digest(args.fit_visual),
            "calibration_visual": digest(args.calibration_visual)},
        "partitions": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({part: output[part]["metrics"] for part in output}, indent=2))


if __name__ == "__main__":
    main()
