"""CPU probe of future progress from cached true group-four policy states.

This reused R2R-train development experiment is exploratory. Oracle
geodesic distances define targets and never enter the frozen encoder
features. No GPU, new Habitat replay, reward, or online RL is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

import torch
from torch.nn import functional as F


FORWARD = re.compile(r"move forward (\d+)cm")
ANCHORS = (3, 6)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_success(manifest: dict, source_root: Path) -> dict[tuple[int, str], list[bool]]:
    requested = {(int(group["seed"]), str(group["episode_id"]))
                 for part in ("fit", "development")
                 for group in manifest["groups"][part]}
    success = defaultdict(list)
    for seed_text, meta in manifest["sources"].items():
        seed = int(seed_text)
        path = source_root / meta["path"]
        if digest(path) != meta["sha256"]:
            raise ValueError(f"source rollout changed: seed {seed}")
        with path.open() as stream:
            for line in stream:
                for info in json.loads(line)["info"]:
                    key = seed, str(info["episode_id"])
                    if key in requested:
                        success[key].append(bool(info["task_success"]))
    if set(success) != requested or any(len(value) != 4
                                         for value in success.values()):
        raise ValueError("incomplete group-four source outcome lookup")
    return success


def forward_meters(record: dict, anchor: int) -> float:
    centimeters = 0
    for turn in record["turns"]:
        if turn["original_turn_index"] > anchor:
            break
        for action in turn["motion_actions"]:
            if action.startswith("move forward"):
                match = FORWARD.fullmatch(action)
                if match is None:
                    raise ValueError(f"unknown forward action: {action}")
                centimeters += int(match.group(1))
    return centimeters / 100.0


def load_part(part: str, manifest: dict, manifest_sha: str,
              turn_root: Path, cache_root: Path, source_id: str,
              success: dict) -> tuple[dict, dict]:
    rows = {anchor: [] for anchor in ANCHORS}
    counts = Counter()
    for group in manifest["groups"][part]:
        seed, eid = int(group["seed"]), str(group["episode_id"])
        if any(success[(seed, eid)]):
            continue
        counts["all_failure_group_ids"] += 1
        group_states = []
        for variant in range(4):
            rid = f"s{seed}_e{eid}_v{variant}"
            record = json.loads((turn_root / part / "records" /
                                 f"{rid}.json").read_text())
            cache = torch.load(cache_root / part / "records" /
                               f"{rid}.pt", map_location="cpu",
                               weights_only=True)
            if record["record_id"] != rid or \
                    record["manifest_sha256"] != manifest_sha or \
                    record["scene_id"] != group["scene_id"] or \
                    cache["schema"] != "group_relative_state_cache_v1" or \
                    cache["manifest_sha256"] != manifest_sha or \
                    cache["source_id"] != source_id or \
                    cache["record_id"] != rid or \
                    cache["hidden"].shape[0] != len(cache["anchor_turns"]):
                raise ValueError(f"record/cache mismatch: {rid}")
            distances = {int(turn["original_turn_index"]):
                         float(turn["distance_to_goal_for_label_only"])
                         for turn in record["turns"]}
            terminal = float(record[
                "replayed_terminal_distance_m_for_audit_only"])
            if not math.isfinite(terminal):
                raise ValueError(f"nonfinite terminal distance: {rid}")
            states = {}
            for index, anchor in enumerate(cache["anchor_turns"]):
                if anchor not in distances:
                    raise ValueError(f"cached anchor absent from record: {rid}")
                vector = cache["hidden"][index].float()
                if vector.shape != (2048,) or not torch.isfinite(vector).all():
                    raise ValueError(f"invalid cached feature: {rid}/{anchor}")
                states[int(anchor)] = (F.normalize(vector, dim=0),
                                       distances[anchor] - terminal,
                                       forward_meters(record, int(anchor)))
            group_states.append((record["terminal_mode"], states))
        for anchor in ANCHORS:
            active = [(mode, states[anchor]) for mode, states in group_states
                      if anchor in states]
            for (lm, (lh, lf, la)), (rm, (rh, rf, ra)) in \
                    itertools.combinations(active, 2):
                gap = lf - rf
                if not math.isfinite(gap) or abs(gap) < .25:
                    continue
                action_gap = la - ra
                rows[anchor].append({
                    "scene": group["scene_id"], "episode_id": eid,
                    "group_id": group["group_id"],
                    "difference": lh - rh,
                    "label": 1.0 if gap > 0 else -1.0,
                    "same_terminal_mode": lm == rm,
                    "action_point": (
                        float((action_gap > 0) == (gap > 0))
                        if action_gap else .5),
                })
    counts["unique_all_failure_episodes"] = len({
        (row["scene"], row["episode_id"])
        for anchor in ANCHORS for row in rows[anchor]})
    return rows, dict(counts)


def fit_linear(rows: list[dict]) -> torch.Tensor:
    # Each episode contributes equal total weight; correlated route pairs
    # do not count as independent observations.
    by_episode = Counter((row["scene"], row["episode_id"]) for row in rows)
    x = torch.stack([row["difference"] for row in rows]).double()
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float64)
    weight = torch.tensor([
        1 / math.sqrt(by_episode[(row["scene"], row["episode_id"])])
        for row in rows], dtype=torch.float64)
    x *= weight[:, None]
    y *= weight
    # Fixed regularization, no development-set tuning or checkpoint choice.
    alpha = 1.0
    dual = torch.linalg.solve(
        x @ x.T + alpha * torch.eye(len(rows), dtype=torch.float64), y)
    return (x.T @ dual).float()


def summarize(rows: list[dict], weight: torch.Tensor) -> dict:
    if not rows:
        return {"pairs": 0, "episode_groups": 0, "scenes": 0,
                "model_episode_macro": None, "action_episode_macro": None}
    by_episode = defaultdict(list)
    for row in rows:
        margin = float(torch.dot(weight, row["difference"])) * row["label"]
        model = 1.0 if margin > 0 else (0.0 if margin < 0 else .5)
        by_episode[(row["scene"], row["episode_id"])].append(
            (model, row["action_point"]))
    by_scene = defaultdict(list)
    for (scene, _), values in by_episode.items():
        by_scene[scene].append(statistics.mean(v[0] for v in values))
    return {
        "pairs": len(rows), "episode_groups": len(by_episode),
        "scenes": len(by_scene),
        "model_episode_macro": statistics.mean(
            statistics.mean(v[0] for v in values)
            for values in by_episode.values()),
        "action_episode_macro": statistics.mean(
            statistics.mean(v[1] for v in values)
            for values in by_episode.values()),
        "model_scene_macro": statistics.mean(
            statistics.mean(values) for values in by_scene.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "turn-root", "cache-root", "cache-audit",
                 "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.cache_audit.read_text())
    manifest_sha = digest(args.manifest)
    if manifest.get("schema") != "policy_group_relative_manifest_v1" or \
            audit.get("schema") != "group_relative_state_cache_audit_v1" or \
            audit.get("manifest_sha256") != manifest_sha:
        raise ValueError("frozen manifest/cache audit mismatch")
    source_id = audit["source_id"]
    success = load_success(manifest, args.source_root)
    fit, fit_counts = load_part("fit", manifest, manifest_sha,
                                args.turn_root, args.cache_root,
                                source_id, success)
    dev, dev_counts = load_part("development", manifest, manifest_sha,
                                args.turn_root, args.cache_root,
                                source_id, success)
    if {row["scene"] for anchor in ANCHORS for row in fit[anchor]} & \
            {row["scene"] for anchor in ANCHORS for row in dev[anchor]}:
        raise ValueError("fit/development scene leakage")
    training = [row for anchor in ANCHORS for row in fit[anchor]
                if row["same_terminal_mode"]]
    if not training:
        raise ValueError("no same-terminal-mode fit pairs")
    weight = fit_linear(training)
    results = {}
    for part, rows, counts in (("fit", fit, fit_counts),
                               ("development", dev, dev_counts)):
        results[part] = {"counts": counts, "anchors": {}}
        for anchor in ANCHORS:
            results[part]["anchors"][str(anchor)] = {
                "all_pairs": summarize(rows[anchor], weight),
                "same_terminal_mode": summarize(
                    [row for row in rows[anchor]
                     if row["same_terminal_mode"]], weight),
            }
    report = {
        "schema": "cached_group4_future_advantage_linear_probe_v1",
        "manifest_sha256": manifest_sha,
        "cache_audit_sha256": digest(args.cache_audit),
        "source_id": source_id,
        "feature": "unit-normalized frozen 2048-dimensional policy-history vector",
        "target": "same-episode future geodesic progress after anchors 3/6",
        "training": "all-failure, same-terminal-mode pairwise weighted ridge; alpha=1; equal episode weight",
        "fit": results["fit"], "development": results["development"],
        "interpretation": (
            "Reused train-scene development probe with correlated episodes "
            "across seeds. No fresh audit, learned reward, online RL, or "
            "val-unseen navigation result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results["development"], indent=2))


if __name__ == "__main__":
    main()
