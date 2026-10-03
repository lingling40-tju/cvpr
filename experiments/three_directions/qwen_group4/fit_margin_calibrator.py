"""Exploratory train-scene calibration of frozen route-match margins.

Fit only the first 40 on-policy steps. A hash-defined scene partition is
reserved for diagnosis. Simulator distance supplies fit labels but is never
an input feature. This is not a navigation evaluation or an online reward.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import itertools

import numpy as np


MODES = {"stopped but goal not reached.", "number of turns exceeded."}
PREFIX_STEPS = 40
HELDOUT_RESIDUE = 2


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def route_features(item: dict) -> np.ndarray:
    teacher = item["fused_reward"]
    if teacher["status"] != "ok":
        raise ValueError("failed route missing teacher score")
    value = np.array([
        float(teacher["first_margin"]),
        float(teacher["swapped_margin"]),
        float(len(item["executed_actions"])),
        float(len(item["gen_traj"])),
    ], dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite route features")
    if abs(float(teacher["raw"]) - (value[0] + value[1]) / 2) > 1e-6:
        raise ValueError("teacher order average mismatch")
    return value


def examples(rollout_path: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "qwen3_group4_exact_start_dataset_v1" or \
            manifest["selected_rows"] != 256:
        raise ValueError("unexpected training manifest")
    scene_by_id = {str(row["episode_id"]): row["scene_id"]
                   for row in manifest["rows"]}
    if len(scene_by_id) != 256:
        raise ValueError("duplicate training episode")
    with rollout_path.open("rb") as stream:
        lines = [stream.readline() for _ in range(PREFIX_STEPS)]
    if any(not line for line in lines):
        raise ValueError("first 40 rollout steps unavailable")
    seen = set()
    rows = []
    for step, line in enumerate(lines, 1):
        record = json.loads(line)
        if record["step"] != step or len(record["info"]) != 16:
            raise ValueError("rollout step or sample count mismatch")
        groups = defaultdict(list)
        for item in record["info"]:
            groups[str(item["episode_id"])].append(item)
        if len(groups) != 4 or set(map(len, groups.values())) != {4}:
            raise ValueError("expected four four-rollout groups per step")
        for eid, group in groups.items():
            if eid in seen or eid not in scene_by_id:
                raise ValueError("training episode reused or absent")
            seen.add(eid)
            if any(item["task_success"] for item in group):
                continue
            scene = scene_by_id[eid]
            for left, right in itertools.combinations(group, 2):
                if left["end_reason"] != right["end_reason"] or \
                        left["end_reason"] not in MODES:
                    continue
                da, db = float(left["distance_to_goal"]), float(right["distance_to_goal"])
                if not math.isfinite(da) or not math.isfinite(db):
                    raise ValueError("nonfinite fit label")
                if abs(da - db) < 1.5:
                    continue
                better, worse = (left, right) if da < db else (right, left)
                rows.append((scene, route_features(better) - route_features(worse)))
    if len(seen) != 160 or not rows:
        raise ValueError("first-40 coverage mismatch")
    return rows, digest_bytes(b"".join(lines))


def fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.std(x, axis=0), 1.0)
    z = x / scale
    theta = np.zeros(z.shape[1], dtype=np.float64)
    ridge = 0.1
    for _ in range(40):
        margin = z @ theta
        wrong = np.exp(-np.logaddexp(0.0, margin))
        gradient = -(z.T @ wrong) / len(z) + ridge * theta
        weights = wrong * (1.0 - wrong)
        hessian = (z.T * weights) @ z / len(z) + ridge * np.eye(z.shape[1])
        update = np.linalg.solve(hessian, gradient)
        theta -= update
        if float(np.max(np.abs(update))) < 1e-9:
            break
    return theta / scale, scale


def accuracy(x: np.ndarray, weight: np.ndarray) -> dict:
    raw = (x[:, 0] + x[:, 1]) / 2
    calibrated = x @ weight
    return {"pairs": len(x),
            "raw_correct": int(np.sum(raw > 0)),
            "calibrated_correct": int(np.sum(calibrated > 0)),
            "raw_ties": int(np.sum(raw == 0)),
            "calibrated_ties": int(np.sum(calibrated == 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, prefix_sha = examples(args.rollout, args.manifest)
    fitting = np.stack([x for scene, x in rows if
                        int(hashlib.sha256(scene.encode()).hexdigest(), 16) % 5 != HELDOUT_RESIDUE])
    heldout = np.stack([x for scene, x in rows if
                        int(hashlib.sha256(scene.encode()).hexdigest(), 16) % 5 == HELDOUT_RESIDUE])
    if len(fitting) < 80 or len(heldout) < 20:
        raise ValueError("scene split lacks pair coverage")
    weight, scale = fit(fitting)
    report = {"schema": "qwen_margin_calibration_train_scene_v1",
              "interpretation": "Exploratory train-scene distance ranking only; no navigation result.",
              "rollout_prefix_steps": PREFIX_STEPS,
              "rollout_prefix_sha256": prefix_sha,
              "manifest_sha256": digest_bytes(args.manifest.read_bytes()),
              "split": "SHA256(scene_id) modulo 5 == 2 held out",
              "features": ["first_margin", "swapped_margin", "executed_action_count",
                           "turn_count"],
              "ridge": 0.1,
              "feature_scale": scale.tolist(),
              "score_weights": weight.tolist(),
              "fit": accuracy(fitting, weight),
              "heldout_scenes": accuracy(heldout, weight)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
