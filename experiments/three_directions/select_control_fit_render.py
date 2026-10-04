"""Select two audited train-fit variants per episode before RGB replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


EXPECTED_IDS_SHA = "d214dc38cf8d4094a6329afd73c080e60adaa0a6b6e30b69cb92f335aecae081"
SALT = "control-fit-uniform-variant-v1|"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(plan: dict) -> str:
    return hashlib.sha256((SALT + str(plan["episode_id"]) + ":" +
                           str(plan["variant"])).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--all-manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.ids) != EXPECTED_IDS_SHA:
        raise ValueError("frozen ID selection changed")
    ids = json.loads(args.ids.read_text())
    all_manifest = json.loads(args.all_manifest.read_text())
    all_sha = digest(args.all_manifest)
    summary = json.loads((args.labels_root / "summary.json").read_text())
    if ids["schema"] != "control_exact512_policy_fit_extension_ids_v1" or \
            all_manifest["schema"] != "policy_process_train_manifest_v1" or \
            all_manifest["source_id_manifest_sha256"] != EXPECTED_IDS_SHA or \
            all_manifest["targets"] != {"fit": 1024,
                                        "development": 0, "audit": 0} or \
            summary["schema"] != "control_fit_label_only_replay_summary_v1" or \
            summary["manifest_sha256"] != all_sha or \
            summary["requested"] != 1024 or \
            summary["completed"] != 1024 or \
            summary["smoke_limit"] or summary["rgb_frames_written"]:
        raise ValueError("label-only replay source incomplete")
    plans_by_episode = defaultdict(list)
    for plan in all_manifest["selected"]["fit"]:
        plans_by_episode[str(plan["episode_id"])].append(plan)
    if len(plans_by_episode) != 256 or \
            any(len(rows) != 4 for rows in plans_by_episode.values()):
        raise ValueError("not 256 complete groups of four")
    chosen = []
    recomputed = {"all_forward": 0, "all_regression": 0,
                  "all_episodes_with_regression": set(),
                  "chosen_forward": 0, "chosen_regression": 0,
                  "chosen_episodes_with_regression": set()}
    for plan_id in ids["rows"]:
        eid, scene = str(plan_id["episode_id"]), str(plan_id["scene_id"])
        rows = plans_by_episode[eid]
        if len({str(row["scene_id"]) for row in rows}) != 1 or \
                str(rows[0]["scene_id"]) != scene:
            raise ValueError(f"scene mismatch for episode {eid}")
        labels = {}
        for plan in rows:
            rid = f"s{plan['seed']}_e{eid}_v{plan['variant']}"
            record = json.loads((args.labels_root / "records" /
                                 f"{rid}.json").read_text())
            distances = record["distance_to_goal_m_for_fit_label_only"]
            if record["schema"] != "control_fit_label_only_replay_v1" or \
                    record["manifest_sha256"] != all_sha or \
                    record["record_id"] != rid or \
                    record["scene_id"] != scene or \
                    record["terminal_mode"] != plan["terminal_mode"] or \
                    abs(record["source_terminal_distance_m"] -
                        plan["terminal_distance_m_for_replay_audit_only"]) > 1e-5 or \
                    abs(record["replayed_terminal_distance_m"] -
                        distances[-1]) > 1e-5 or \
                    abs(distances[-1] -
                        plan["terminal_distance_m_for_replay_audit_only"]) > .25 or \
                    not all(math.isfinite(d) and d >= 0 for d in distances):
                raise ValueError(f"invalid label-only record {rid}")
            forward = sum(a - b >= 1.0 for a, b in
                          zip(distances[:-1], distances[1:]))
            backward = sum(b - a >= 1.0 for a, b in
                           zip(distances[:-1], distances[1:]))
            if forward != record["forward_turns_1m"] or \
                    backward != record["regression_turns_1m"]:
                raise ValueError(f"geodesic count mismatch {rid}")
            labels[int(plan["variant"])] = (forward, backward)
            recomputed["all_forward"] += forward
            recomputed["all_regression"] += backward
            if backward:
                recomputed["all_episodes_with_regression"].add(eid)
        uniform = min(rows, key=rank)
        enriched = min((row for row in rows if row != uniform),
                       key=lambda row: (-labels[int(row["variant"])][1],
                                        rank(row)))
        for plan in (uniform, enriched):
            forward, backward = labels[int(plan["variant"])]
            chosen.append(plan)
            recomputed["chosen_forward"] += forward
            recomputed["chosen_regression"] += backward
            if backward:
                recomputed["chosen_episodes_with_regression"].add(eid)
    if recomputed["all_forward"] != summary["forward_turns_1m"] or \
            recomputed["all_regression"] != summary["regression_turns_1m"] or \
            len(recomputed["all_episodes_with_regression"]) != \
            summary["episode_ids_with_regression"] or \
            len(chosen) != 512:
        raise ValueError("label summary or selected variant count differs")
    gate = ids["fit_only_sample_gate"]
    passed = (recomputed["chosen_regression"] >=
              gate["minimum_one_meter_regressions"] and
              len(recomputed["chosen_episodes_with_regression"]) >=
              gate["minimum_episode_ids_with_regression"])
    report = {
        "schema": "control_fit_render_selection_v1",
        "interpretation": "Fit-only geodesic label coverage; no learned reward or navigation result.",
        "source_sha256": {"ids": EXPECTED_IDS_SHA,
                          "all_variant_manifest": all_sha,
                          "label_only_summary": digest(args.labels_root / "summary.json")},
        "all_four": {"trajectories": 1024,
                     "forward_turns_1m": recomputed["all_forward"],
                     "regression_turns_1m": recomputed["all_regression"],
                     "episode_ids_with_regression": len(recomputed["all_episodes_with_regression"])},
        "chosen": {"trajectories": 512, "episode_ids": 256,
                   "forward_turns_1m": recomputed["chosen_forward"],
                   "regression_turns_1m": recomputed["chosen_regression"],
                   "episode_ids_with_regression": len(recomputed["chosen_episodes_with_regression"])},
        "sample_gate": {**gate, "passed": passed},
        "render_manifest_sha256": None,
    }
    if passed:
        render = {**all_manifest,
                  "selection": "one SHA-uniform and one distinct max-regression variant per frozen fit episode, after complete label-only replay",
                  "source_all_variant_manifest_sha256": all_sha,
                  "targets": {"fit": 512, "development": 0, "audit": 0},
                  "inventory": {"fit": {"episode_ids": 256,
                                          "trajectories": 512,
                                          "scenes": 38}},
                  "selected": {"fit": chosen, "development": [], "audit": []}}
        args.render_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.render_manifest.write_text(json.dumps(render, indent=2) + "\n")
        report["render_manifest_sha256"] = digest(args.render_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
