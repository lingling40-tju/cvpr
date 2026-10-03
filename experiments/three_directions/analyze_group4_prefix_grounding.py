"""Post-hoc diagnosis of already opened prefix-grounding development states.

This report is descriptive. It never changes the frozen candidate gates,
opens the model audit, or uses val-unseen data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest


def score_states(rows: list[dict], root: Path, source: str,
                 manifest_sha: str, scale: torch.Tensor,
                 vector: torch.Tensor) -> dict[tuple[str, int], float]:
    result = {}
    for row in rows:
        eid = str(row["episode_id"])
        path = root / "development" / "records" / f"{eid}.pt"
        item = torch.load(path, map_location="cpu", weights_only=True)
        anchors = [a for a in (3, 6) if a < row["turn_count"]]
        if item["schema"] != "group4_expert_prefix_state_v1" or \
                item["episode_id"] != eid or \
                item["manifest_sha256"] != manifest_sha or \
                item["record_sha256"] != row["record_sha256"] or \
                item["source_id"] != source or item["anchors"] != anchors:
            raise ValueError(f"cache provenance mismatch: {path}")
        correct, wrong = item["correct"].float(), item["wrong"].float()
        if correct.shape != wrong.shape or correct.shape != (len(anchors), 2048):
            raise ValueError(f"cache shape mismatch: {path}")
        for j, anchor in enumerate(anchors):
            diff = F.normalize((correct[j] - wrong[j]) / scale, dim=0)
            margin = float(diff @ vector)
            if not bool(torch.isfinite(torch.tensor(margin))):
                raise ValueError(f"nonfinite margin: {path}")
            result[(eid, anchor)] = margin
    return result


def summarize(rows: list[dict], scores: dict[tuple[str, int], float],
              evidence: dict[tuple[str, int], str]) -> dict:
    anchor_rows = defaultdict(list)
    scene_rows = defaultdict(list)
    evidence_rows = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        eid = str(row["episode_id"])
        for anchor in (3, 6):
            if (eid, anchor) not in scores:
                continue
            passed = int(scores[eid, anchor] > 0)
            anchor_rows[anchor].append(passed)
            scene_rows[(row["scene_id"], anchor)].append(passed)
            evidence_rows[(anchor, evidence[eid, anchor])].append(passed)
            paired[eid][anchor] = passed
    by_anchor = {}
    for anchor in (3, 6):
        values = anchor_rows[anchor]
        scenes = [v for (scene, a), v in scene_rows.items() if a == anchor]
        by_anchor[str(anchor)] = {
            "pairs": len(values), "correct": sum(values),
            "accuracy": sum(values) / len(values),
            "scenes": len(scenes),
            "scene_macro_accuracy": sum(sum(v) / len(v) for v in scenes) / len(scenes),
        }
    both = [v for v in paired.values() if set(v) == {3, 6}]
    by_evidence = {}
    for (anchor, status), values in sorted(evidence_rows.items()):
        by_evidence[f"{anchor}:{status}"] = {
            "pairs": len(values), "correct": sum(values),
            "accuracy": sum(values) / len(values),
        }
    return {"by_anchor": by_anchor, "by_evidence": by_evidence,
            "paired_trajectories": len(both),
            "paired_outcomes": {
                "both_correct": sum(v[3] and v[6] for v in both),
                "early_only": sum(v[3] and not v[6] for v in both),
                "late_only": sum(not v[3] and v[6] for v in both),
                "both_wrong": sum(not v[3] and not v[6] for v in both),
            }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-audit", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--initial-root", type=Path, required=True)
    parser.add_argument("--prefix-root", type=Path, required=True)
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--crossed-root", type=Path, required=True)
    parser.add_argument("--crossed-checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    path_audit = json.loads(args.path_audit.read_text())
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    if manifest["schema"] != "group4_joint_value_expert_manifest_v1" or \
            weights["schema"] != "group4_joint_value_weights_v1":
        raise ValueError("incorrect source schema")
    rows = manifest["selected"]["development"]
    if len(rows) != 161 or len({r["scene_id"] for r in rows}) != 8:
        raise ValueError("development partition changed")
    if set(manifest["selected"]["fit"][i]["scene_id"]
           for i in range(len(manifest["selected"]["fit"]))) & \
            {r["scene_id"] for r in rows}:
        raise ValueError("fit/development scene leakage")
    scale, vector = weights["scale"].float(), weights["vector"].float()
    if scale.shape != vector.shape or scale.shape != (2048,):
        raise ValueError("readout shape mismatch")
    manifest_sha = digest(args.manifest)
    if path_audit["schema"] != "group4_expert_path_ambiguity_audit_v1" or \
            path_audit["manifest_sha256"] != manifest_sha:
        raise ValueError("path audit source mismatch")
    evidence = {}
    for row in path_audit["partitions"]["development"]["rows"]:
        for anchor, status in row["anchor_status"].items():
            evidence[row["episode_id"], int(anchor)] = status
    if len(evidence) != 303:
        raise ValueError("incomplete development evidence labels")
    model_sources = {
        "initial": (args.initial_root, weights["encoder_source_id"]),
        "prefix": (args.prefix_root, digest(args.prefix_checkpoint)),
        "crossed": (args.crossed_root, digest(args.crossed_checkpoint)),
        "temporal": (args.temporal_root, digest(args.temporal_checkpoint)),
    }
    models = {}
    for name, (root, source) in model_sources.items():
        scores = score_states(rows, root, source, manifest_sha, scale, vector)
        if len(scores) != 303:
            raise ValueError(f"incomplete prefix scores for {name}")
        models[name] = summarize(rows, scores, evidence)
    result = {
        "schema": "group4_prefix_grounding_posthoc_v1",
        "interpretation": "already opened train-scene development; descriptive only",
        "source_sha256": {
            "manifest": manifest_sha, "path_audit": digest(args.path_audit),
            "weights": digest(args.weights),
            "prefix_checkpoint": digest(args.prefix_checkpoint),
            "crossed_checkpoint": digest(args.crossed_checkpoint),
            "temporal_checkpoint": digest(args.temporal_checkpoint),
        },
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
