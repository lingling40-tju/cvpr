"""Audit exact policy-prefix route grounding and frozen go/no-go gate."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.policy_manifest.read_text())
    expert = json.loads(args.expert_analysis.read_text())
    if manifest["schema"] != "qwen3_policy_route_manifest_v1" or \
            expert["schema"] != "qwen3_route_match_analysis_v1" or \
            not expert["gate"]["passed"]:
        raise ValueError("unverified frozen teacher or policy source")
    expected = manifest["selected"]["development"]
    if manifest["inventory"]["development"]["groups"] != 13 or \
            manifest["inventory"]["development"]["preterminal_t6"] != 48 or \
            manifest["inventory"]["development"]["success_t6"] != 8 or \
            manifest["inventory"]["development"]["success_failure_pairs_t6"] != 15:
        raise ValueError("frozen label-only coverage changed")
    shards = []
    shard_hashes = {}
    for shard in range(4):
        path = args.shard_root / f"policy_development_shard{shard}.json"
        data = json.loads(path.read_text())
        if data["schema"] != "qwen3_policy_route_shard_v1" or \
                data["part"] != "development" or data["smoke_only"] or \
                data["shard"] != shard or data["shards"] != 4 or \
                data["group_count"] != len(expected[shard::4]) or \
                data["source_sha256"]["policy_manifest"] != digest(args.policy_manifest) or \
                data["source_sha256"]["expert_analysis"] != digest(args.expert_analysis) or \
                data["teacher_prompt_sha256"] != expert["source_sha256"]["prompt"]:
            raise ValueError(f"incomplete or inconsistent policy shard {shard}")
        if [(x["seed"], x["episode_id"]) for x in data["groups"]] != \
                [(x["seed"], x["episode_id"]) for x in expected[shard::4]]:
            raise ValueError(f"group identity changed shard {shard}")
        shards.append(data)
        shard_hashes[str(shard)] = digest(path)
    source = shards[0]["source_sha256"]
    if any(shard["source_sha256"] != source for shard in shards):
        raise ValueError("model/source hash differs between shards")
    route_count = 0
    states = {0: 0, 3: 0, 6: 0}
    success_t6 = []
    success_gain_t3_t6 = []
    success_gain_t0_t6 = []
    all_t6 = []
    failure_t6 = []
    order_consistency_t6 = []
    terminal_modes = Counter()
    group_rank_rows = []
    by_scene = defaultdict(list)
    by_group = defaultdict(list)
    per_group = []
    for shard in shards:
        for group in shard["groups"]:
            expected_group = next(e for e in expected if e["seed"] == group["seed"] and
                                  e["episode_id"] == group["episode_id"])
            if group["scene_id"] != expected_group["scene_id"] or \
                    len(group["routes"]) != 4:
                raise ValueError("scene or group size mismatch")
            output_group = {"seed": group["seed"], "episode_id": group["episode_id"],
                            "scene_id": group["scene_id"], "routes": []}
            t6_success, t6_failure = [], []
            for route, plan in zip(group["routes"], expected_group["routes"]):
                if route["record_id"] != plan["record_id"] or \
                        route["variant"] != plan["variant"] or \
                        route["success_for_analysis_only"] != plan["success_for_analysis_only"] or \
                        [state["turn"] for state in route["states"]] != [0] + plan["anchors"]:
                    raise ValueError("route/anchor selection mismatch")
                margins = {}
                for state in route["states"]:
                    turn = state["turn"]
                    expected_indices = [int(i * turn / 5 + 0.5) for i in range(6)]
                    if state["sampled_frame_indices"] != expected_indices:
                        raise ValueError("policy frame selection changed")
                    avg = state["correct_margin_average"]
                    if abs(avg - (state["correct_margin_first"] +
                                  state["correct_margin_swapped"]) / 2) > 1e-5:
                        raise ValueError("order-averaged margin mismatch")
                    margins[turn] = avg
                    states[turn] += 1
                    if turn == 6:
                        order_consistency_t6.append(int(state["correct_margin_first"] *
                                                        state["correct_margin_swapped"] > 0))
                route_count += 1
                terminal_modes[route["terminal_mode_for_analysis_only"]] += 1
                if 6 in margins:
                    all_t6.append(int(margins[6] > 0))
                    if route["success_for_analysis_only"]:
                        success_t6.append(int(margins[6] > 0))
                        t6_success.append((route["record_id"], margins[6]))
                        if 3 in margins:
                            success_gain_t3_t6.append(int(margins[6] > margins[3]))
                        success_gain_t0_t6.append(int(margins[6] > margins[0]))
                    else:
                        failure_t6.append(int(margins[6] > 0))
                        t6_failure.append((route["record_id"], margins[6]))
                output_group["routes"].append({"record_id": route["record_id"],
                                               "success": route["success_for_analysis_only"],
                                               "terminal_mode": route["terminal_mode_for_analysis_only"],
                                               "margins": margins})
            for success_id, success_score in t6_success:
                for failure_id, failure_score in t6_failure:
                    correct = int(success_score > failure_score)
                    key = f"{group['seed']}:{group['episode_id']}"
                    group_rank_rows.append({"group": key, "scene": group["scene_id"],
                                            "success_record": success_id,
                                            "failure_record": failure_id,
                                            "margin_difference": success_score - failure_score,
                                            "correct": correct})
                    by_scene[group["scene_id"]].append(correct)
                    by_group[key].append(correct)
            per_group.append(output_group)
    if route_count != 52 or states != {0: 52, 3: 51, 6: 48} or \
            len(success_t6) != 8 or len(group_rank_rows) != 15 or \
            len(success_gain_t3_t6) != 8:
        raise ValueError("policy state/opportunity coverage mismatch")
    rng = random.Random(11)
    groups = list(by_group.values())
    boots = []
    for _ in range(2000):
        sample = [rng.choice(groups) for _ in groups]
        boots.append(sum(map(sum, sample)) / sum(map(len, sample)))
    boots.sort()
    metrics = {"success_t6_positive": sum(success_t6), "success_t6_count": 8,
               "success_t3_to_t6_gain": sum(success_gain_t3_t6),
               "success_t0_to_t6_gain": sum(success_gain_t0_t6),
               "success_failure_correct": sum(r["correct"] for r in group_rank_rows),
               "success_failure_pairs": 15,
               "outcome_rank_group_macro": sum(sum(v) / len(v) for v in by_group.values()) / len(by_group),
               "outcome_rank_scene_macro": sum(sum(v) / len(v) for v in by_scene.values()) / len(by_scene),
               "outcome_rank_group_bootstrap95": [boots[50], boots[1949]],
               "all_t6_instruction_positive": sum(all_t6), "all_t6_count": 48,
               "failure_t6_instruction_positive": sum(failure_t6),
               "failure_t6_count": len(failure_t6),
               "t6_order_agreement": sum(order_consistency_t6),
               "t6_order_agreement_count": len(order_consistency_t6),
               "terminal_modes": dict(terminal_modes)}
    passed = metrics["success_t6_positive"] >= 7 and \
             metrics["success_failure_correct"] >= 10 and \
             metrics["success_t3_to_t6_gain"] >= 5
    report = {"schema": "qwen3_policy_route_analysis_v1",
              "interpretation": "exploratory R2R-train policy-history diagnostic; not navigation improvement",
              "source_sha256": {**source, "shards": shard_hashes},
              "coverage": {"groups": 13, "routes": 52, "states_by_turn": states,
                           "group_success_failure_pairs_t6": 15,
                           "mixed_groups_t6": len(by_group), "mixed_scenes_t6": len(by_scene)},
              "metrics": metrics,
              "gate": {"success_positive_min": 7, "outcome_pairs_min": 10,
                       "success_gain_t3_t6_min": 5, "passed": passed},
              "group_rank_rows": group_rank_rows, "per_group": per_group}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coverage": report["coverage"], "metrics": metrics,
                      "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
