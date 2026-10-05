"""Frozen text-and-motion shortcut baseline for boundary occupancy.

This baseline deliberately never opens RGB files. It fits only on the
preassigned fit scenes and evaluates the same frozen development records.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix


HASH_DIM = 8192
FEATURE_DIM = HASH_DIM + 16
L2 = .001


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bucket(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(),
                          "little") % HASH_DIM


def features(instruction: str, history: list[dict]) -> dict[int, float]:
    result = defaultdict(float)
    words = re.findall(r"[a-z0-9]+", instruction.lower())
    if not words or len(words) > 300:
        raise ValueError("bad natural instruction")
    for word in words:
        result[bucket("w:" + word)] += 1 / len(words)
    for left, right in zip(words, words[1:]):
        result[bucket("b:" + left + "_" + right)] += 1 / max(1, len(words)-1)
    recent = history[-4:]
    motions = [a for turn in recent for a in turn["executed_actions"]]
    forward = left = right = 0.0
    for action in motions:
        result[bucket("a:" + action)] += .25
        if action.startswith("move forward "):
            forward += int(action.split()[-1][:-2]) / 100
        elif action.startswith("turn left "):
            left += int(action.split()[2]) / 90
        elif action.startswith("turn right "):
            right += int(action.split()[2]) / 90
        else:
            raise ValueError(f"unknown motion action: {action}")
    numeric = [1.0, len(recent)/4, len(motions)/12, forward/5, left/4,
               right/4, (right-left)/4, int(bool(motions)),
               int(bool(motions) and motions[-1].startswith("move")),
               int(bool(motions) and motions[-1].startswith("turn left")),
               int(bool(motions) and motions[-1].startswith("turn right")),
               len(words)/100]
    for index, value in enumerate(numeric):
        result[HASH_DIM + index] = value
    return result


def load_rows(part: str, manifest: dict, replay: Path, manifest_sha: str,
              labels: dict) -> list[dict]:
    if part not in ("fit", "development"):
        raise ValueError("audit scenes must stay unopened")
    label_by_id = {x["record_id"]: x for x in labels["selected"][part]}
    rows = []
    for plan in manifest["selected"][part]:
        rid = plan["record_id"]
        record = json.loads((replay / part / "records" / f"{rid}.json").read_text())
        if record["manifest_sha256"] != manifest_sha or \
                record["record_id"] != rid or rid not in label_by_id or \
                plan["scene_id"] not in manifest["scene_split"][part]:
            raise ValueError(f"changed frozen shortcut input: {rid}")
        inp = record["input"]
        for cls, state, instruction in (
                ("outside", "outside", inp["instruction"]),
                ("inside", "inside", inp["instruction"]),
                ("wrong", "inside", inp["wrong_instruction"])):
            if instruction is None or part == "fit" and cls == "wrong" and \
                    not plan["wrong_instruction_same_start"]:
                continue
            rows.append({"record_id": rid, "episode_id": str(plan["episode_id"]),
                         "scene_id": plan["scene_id"], "class": cls,
                         "task_success": label_by_id[rid]["task_success"],
                         "same_start_wrong": plan["wrong_instruction_same_start"],
                         "features": features(
                             instruction,
                             inp["action_history_by_state"][state])})
    if not rows:
        raise ValueError(f"empty {part} shortcut rows")
    return rows


def matrix(rows: list[dict]) -> csr_matrix:
    positions = [(i, col, value) for i, row in enumerate(rows)
                 for col, value in row["features"].items()]
    rr, cc, vv = zip(*positions)
    return csr_matrix((vv, (rr, cc)), shape=(len(rows), FEATURE_DIM),
                      dtype=np.float64)


def threshold_for(positive: np.ndarray, negative: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate([positive, negative]))[::-1]
    feasible = []
    for value in np.concatenate([[candidates[0] + 1], candidates]):
        fpr = float(np.mean(negative >= value))
        if fpr <= .05:
            feasible.append((float(np.mean(positive >= value)), -fpr,
                             float(value)))
    recall, neg_fpr, value = max(feasible)
    return {"threshold": value, "pooled_recall": recall,
            "pooled_fpr": -neg_fpr}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    delta = pos[:, None] - neg[None, :]
    return float(np.mean((delta > 0) + .5 * (delta == 0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "verification", "replay-root",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest, labels, verification = [json.loads(p.read_text()) for p in
                                      (args.manifest, args.labels,
                                       args.verification)]
    manifest_sha = digest(args.manifest)
    if labels["replay_manifest_sha256"] != manifest_sha or \
            verification["manifest_sha256"] != manifest_sha or \
            verification["smoke_record_id"] is not None or \
            [p["records"] for p in verification["parts"]] != [1184, 288]:
        raise ValueError("unverified frozen source")
    fit = load_rows("fit", manifest, args.replay_root, manifest_sha, labels)
    dev = load_rows("development", manifest, args.replay_root, manifest_sha,
                    labels)
    x, y = matrix(fit), np.array([r["class"] == "inside" for r in fit],
                                 dtype=np.float64)
    def objective(weights):
        logits = x @ weights
        loss = np.mean(np.logaddexp(0, logits) - y * logits) + \
               .5 * L2 * float(weights @ weights)
        gradient = np.asarray(x.T @ (expit(logits)-y)).ravel()/len(y) + \
                   L2*weights
        return float(loss), gradient
    result = minimize(objective, np.zeros(FEATURE_DIM), jac=True,
                      method="L-BFGS-B", options={"maxiter": 300,
                                                "ftol": 1e-10})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"shortcut fit failed: {result.message}")
    scores = matrix(dev) @ result.x
    positive = scores[[r["class"] == "inside" for r in dev]]
    negative = scores[[r["class"] != "inside" for r in dev]]
    threshold = threshold_for(positive, negative)
    by_class = {kind: [float(scores[i]) for i, row in enumerate(dev)
                       if row["class"] == kind]
                for kind in ("outside", "inside", "wrong")}
    by_id = defaultdict(dict)
    for index, row in enumerate(dev):
        by_id[row["record_id"]][row["class"]] = float(scores[index])
    t = threshold["threshold"]
    near_failure = [float(scores[i]) for i, row in enumerate(dev) if
                    row["class"] == "inside" and not row["task_success"]]
    report = {"schema": "boundary_history_only_shortcut_v1",
              "manifest_sha256": manifest_sha,
              "labels_sha256": digest(args.labels),
              "verification_sha256": digest(args.verification),
              "image_files_opened": 0,
              "feature_dim": FEATURE_DIM, "l2": L2,
              "fit_rows": len(fit), "development_rows": len(dev),
              "fit_scene_count": len({r["scene_id"] for r in fit}),
              "development_scene_count": len({r["scene_id"] for r in dev}),
              "optimizer_iterations": int(result.nit),
              "auc_pooled": auc(positive, negative),
              "threshold": threshold,
              "outside_fpr": float(np.mean(np.array(by_class["outside"]) >= t)),
              "wrong_instruction_fpr": float(np.mean(
                  np.array(by_class["wrong"]) >= t)),
              "near_failure_recall": float(np.mean(np.array(near_failure) >= t)),
              "crossing_order_accuracy": sum(v["inside"] > v["outside"]
                                             for v in by_id.values())/len(by_id),
              "instruction_order_accuracy": sum(v["inside"] > v["wrong"]
                                                for v in by_id.values() if
                                                "wrong" in v)/len(by_class["wrong"]),
              "navigation_result": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
