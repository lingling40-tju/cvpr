"""Audit frozen terminal route-fidelity teacher against group-four outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--expert-analysis", type=Path, required=True)
    parser.add_argument("--part", choices=("fit", "development"), default="development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.policy_manifest.read_text())
    expert = json.loads(args.expert_analysis.read_text())
    if manifest["schema"] != "qwen3_policy_route_manifest_v1" or \
            expert["schema"] != "qwen3_route_match_analysis_v1" or \
            not expert["gate"]["passed"]:
        raise ValueError("unverified frozen sources")
    expected = manifest["selected"][args.part]
    expected_groups = {"fit": 55, "development": 13}[args.part]
    expected_success = {"fit": 39, "development": 11}[args.part]
    expected_pairs = {"fit": 71, "development": 19}[args.part]
    expected_mixed = {"fit": 21, "development": 6}[args.part]
    if len(expected) != expected_groups or \
            sum(r["success_for_analysis_only"] for g in expected for r in g["routes"]) != expected_success:
        raise ValueError("terminal source labels changed")
    shard_hashes = {}
    shards = []
    for shard in range(4):
        path = args.shard_root / f"terminal_{args.part}_shard{shard}.json"
        data = json.loads(path.read_text())
        if data["schema"] != "qwen3_policy_terminal_shard_v1" or \
                data["part"] != args.part or data["smoke_only"] or \
                data["shard"] != shard or data["shards"] != 4 or \
                data["group_count"] != len(expected[shard::4]) or \
                data["source_sha256"]["policy_manifest"] != digest(args.policy_manifest) or \
                data["source_sha256"]["expert_analysis"] != digest(args.expert_analysis) or \
                data["teacher_prompt_sha256"] != expert["source_sha256"]["prompt"]:
            raise ValueError(f"incomplete terminal shard {shard}")
        if [(g["seed"], g["episode_id"]) for g in data["groups"]] != \
                [(g["seed"], g["episode_id"]) for g in expected[shard::4]]:
            raise ValueError("group identity changed")
        shards.append(data)
        shard_hashes[str(shard)] = digest(path)
    source = shards[0]["source_sha256"]
    if any(shard["source_sha256"] != source for shard in shards):
        raise ValueError("source/model hashes differ across shards")
    success_positive = []
    success_gain = []
    all_positive = []
    failure_positive = []
    order_consistency = []
    terminal_modes = Counter()
    paired = []
    by_group = defaultdict(list)
    by_scene = defaultdict(list)
    per_group = []
    route_count = 0
    for shard in shards:
        for group in shard["groups"]:
            plan = next(e for e in expected if e["seed"] == group["seed"] and
                        e["episode_id"] == group["episode_id"])
            if group["scene_id"] != plan["scene_id"] or len(group["routes"]) != 4:
                raise ValueError("scene or group-size mismatch")
            output_group = {"seed": group["seed"], "episode_id": group["episode_id"],
                            "scene_id": group["scene_id"], "routes": []}
            successes, failures = [], []
            for route, expected_route in zip(group["routes"], plan["routes"]):
                if route["record_id"] != expected_route["record_id"] or \
                        route["variant"] != expected_route["variant"] or \
                        route["success_for_analysis_only"] != expected_route["success_for_analysis_only"] or \
                        [state["turn"] for state in route["states"]] != [0, expected_route["turns"]]:
                    raise ValueError("route or terminal anchor mismatch")
                initial, terminal = route["states"]
                for state in (initial, terminal):
                    turn = state["turn"]
                    if state["sampled_frame_indices"] != \
                            [int(i * turn / 5 + 0.5) for i in range(6)] or \
                            abs(state["correct_margin_average"] -
                                (state["correct_margin_first"] +
                                 state["correct_margin_swapped"]) / 2) > 1e-5:
                        raise ValueError("sampling or A/B averaging mismatch")
                init_score = initial["correct_margin_average"]
                terminal_score = terminal["correct_margin_average"]
                order_consistency.append(int(terminal["correct_margin_first"] *
                                             terminal["correct_margin_swapped"] > 0))
                all_positive.append(int(terminal_score > 0))
                route_count += 1
                terminal_modes[route["terminal_mode_for_analysis_only"]] += 1
                if route["success_for_analysis_only"]:
                    success_positive.append(int(terminal_score > 0))
                    success_gain.append(int(terminal_score > init_score))
                    successes.append((route["record_id"], terminal_score))
                else:
                    failure_positive.append(int(terminal_score > 0))
                    failures.append((route["record_id"], terminal_score))
                output_group["routes"].append({"record_id": route["record_id"],
                                               "success": route["success_for_analysis_only"],
                                               "terminal_mode": route["terminal_mode_for_analysis_only"],
                                               "initial_margin": init_score,
                                               "terminal_margin": terminal_score,
                                               "terminal_turn": terminal["turn"]})
            for sid, sscore in successes:
                for fid, fscore in failures:
                    correct = int(sscore > fscore)
                    key = f"{group['seed']}:{group['episode_id']}"
                    paired.append({"group": key, "scene": group["scene_id"],
                                   "success_record": sid, "failure_record": fid,
                                   "score_difference": sscore - fscore,
                                   "correct": correct})
                    by_group[key].append(correct)
                    by_scene[group["scene_id"]].append(correct)
            per_group.append(output_group)
    if route_count != 4 * expected_groups or len(success_positive) != expected_success or \
            len(paired) != expected_pairs or len(by_group) != expected_mixed:
        raise ValueError("terminal opportunity coverage mismatch")
    rng = random.Random(11)
    clusters = list(by_group.values())
    boots = []
    for _ in range(2000):
        sample = [rng.choice(clusters) for _ in clusters]
        boots.append(sum(map(sum, sample)) / sum(map(len, sample)))
    boots.sort()
    metrics = {"success_terminal_positive": sum(success_positive),
               "success_terminal_count": expected_success,
               "success_terminal_gain_over_initial": sum(success_gain),
               "outcome_pair_correct": sum(x["correct"] for x in paired),
               "outcome_pairs": expected_pairs,
               "outcome_rank_group_macro": sum(sum(v) / len(v) for v in by_group.values()) / len(by_group),
               "outcome_rank_scene_macro": sum(sum(v) / len(v) for v in by_scene.values()) / len(by_scene),
               "outcome_rank_group_bootstrap95": [boots[50], boots[1949]],
               "all_terminal_instruction_positive": sum(all_positive),
               "all_terminal_count": route_count,
               "failure_terminal_instruction_positive": sum(failure_positive),
               "failure_terminal_count": len(failure_positive),
               "terminal_order_agreement": sum(order_consistency),
               "terminal_order_agreement_count": len(order_consistency),
               "terminal_modes": dict(terminal_modes)}
    if args.part == "fit":
        passed = metrics["outcome_pair_correct"] >= 48 and \
                 metrics["outcome_rank_group_macro"] >= 0.65
        gate = {"outcome_pair_min": 48, "group_macro_min": 0.65, "passed": passed}
    else:
        passed = metrics["success_terminal_positive"] >= 9 and \
                 metrics["outcome_pair_correct"] >= 13 and \
                 metrics["success_terminal_gain_over_initial"] >= 7
        gate = {"success_positive_min": 9, "outcome_pair_min": 13,
                "success_gain_min": 7, "passed": passed}
    report = {"schema": "qwen3_terminal_route_analysis_v1", "part": args.part,
              "interpretation": "exploratory train-scene terminal route matching; no online reward or navigation result",
              "source_sha256": {**source, "shards": shard_hashes},
              "coverage": {"groups": expected_groups, "rollouts": route_count,
                           "terminal_states": route_count, "initial_states": route_count,
                           "outcome_pairs": expected_pairs, "mixed_groups": expected_mixed,
                           "mixed_scenes": len(by_scene)},
              "metrics": metrics,
              "gate": gate,
              "paired_rows": paired, "per_group": per_group}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coverage": report["coverage"], "metrics": metrics,
                      "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
