"""Apply predeclared evidence-onset development gates without audit access."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest, load_part, metrics
from fit_group4_expert_prefix_value import expert_data, expert_metrics


def score_rows(root: Path, kind: str, checkpoint_sha: str,
               manifest_sha: str, weights_sha: str,
               expected: list[dict]) -> list[dict]:
    rows = []
    for shard in range(3):
        path = root / f"shard{shard}.json"
        payload = json.loads(path.read_text())
        selection = expected[shard::3]
        if payload["schema"] != "group4_evidence_onset_dev_score_shard_v1" or \
                payload["source_kind"] != kind or \
                payload["shard"] != shard or payload["shards"] != 3 or \
                payload["manifest_sha256"] != manifest_sha or \
                payload["checkpoint_sha256"] != checkpoint_sha or \
                payload["frozen_weights_sha256"] != weights_sha or \
                payload["pairs"] != len(selection) or \
                len(payload["rows"]) != len(selection):
            raise ValueError(f"incomplete evidence score shard {kind}/{shard}")
        for row, plan in zip(payload["rows"], selection):
            if row["episode_a"] != plan["episode_a"] or \
                    row["episode_b"] != plan["episode_b"] or \
                    row["scene_id"] != plan["scene_id"] or \
                    row["anchor"] != plan["anchor"] or \
                    row["evidence_onset_eligible"] != plan["evidence_onset_eligible"] or \
                    ("onset_delta_a" in row) != plan["evidence_onset_eligible"] or \
                    ("onset_delta_b" in row) != plan["evidence_onset_eligible"]:
                raise ValueError(f"evidence score source mismatch {kind}/{shard}")
            rows.append(row)
    if len(rows) != len(expected):
        raise ValueError(f"incomplete evidence scores {kind}")
    return rows


def summarize_cross(rows: list[dict]) -> dict:
    by_scene = defaultdict(list)
    by_anchor = defaultdict(list)
    onset_by_scene = defaultdict(list)
    onset = []
    individual = []
    for row in rows:
        comparisons = [int(row[k] > 0) for k in
                       ("row_a", "row_b", "column_a", "column_b")]
        individual.extend(comparisons)
        passed = int(all(comparisons))
        by_scene[row["scene_id"]].append(passed)
        by_anchor[row["anchor"]].append(passed)
        if row["evidence_onset_eligible"]:
            for key in ("onset_delta_a", "onset_delta_b"):
                positive = int(row[key] > 0)
                onset.append(positive)
                onset_by_scene[row["scene_id"]].append(positive)
    total = sum(map(len, by_scene.values()))
    if total != 94 or len(onset) != 88:
        raise ValueError("evidence score coverage mismatch")
    correct = sum(map(sum, by_scene.values()))
    return {
        "cross": {"pairs": total, "individual_comparisons": len(individual),
                  "individual_correct": sum(individual),
                  "individual_accuracy": sum(individual) / len(individual),
                  "all_four_correct": correct,
                  "all_four_accuracy": correct / total,
                  "scenes": len(by_scene),
                  "scene_macro_accuracy": sum(sum(v) / len(v)
                                              for v in by_scene.values()) / len(by_scene),
                  "by_anchor": {str(a): {"pairs": len(v), "correct": sum(v),
                                         "accuracy": sum(v) / len(v)}
                                for a, v in sorted(by_anchor.items())}},
        "onset": {"pairs": len(onset) // 2, "route_margins": len(onset),
                  "positive": sum(onset), "positive_rate": sum(onset) / len(onset),
                  "scenes": len(onset_by_scene),
                  "scene_macro_positive_rate": sum(sum(v) / len(v)
                                                   for v in onset_by_scene.values()) /
                                               len(onset_by_scene)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--group-state-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--expert-state-root", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--initial-score-root", type=Path, required=True)
    parser.add_argument("--candidate-score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group, expert, evidence = (json.loads(p.read_text()) for p in
                               (args.group_manifest, args.expert_manifest,
                                args.evidence_manifest))
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    frozen = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    audit = json.loads(args.cache_audit.read_text())
    training = json.loads(args.training_report.read_text())
    group_sha, expert_sha, evidence_sha = (digest(p) for p in
                                          (args.group_manifest, args.expert_manifest,
                                           args.evidence_manifest))
    candidate_sha, initial_sha = digest(args.checkpoint), digest(args.initial_checkpoint)
    weights_sha = digest(args.frozen_weights)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            evidence["schema"] != "group4_evidence_onset_manifest_v1" or \
            evidence["source_expert_manifest_sha256"] != expert_sha or \
            evidence_sha != "41b465132819ae6060d06b3cf0a6ff964d9be83d3ace265ae39414aa1258f26f" or \
            trained["schema"] != "group4_evidence_onset_lora_v1" or \
            training["schema"] != "group4_evidence_onset_lora_training_v1" or \
            training["smoke"] or training["microsteps"] != 384 or \
            training["model_forward_count"] != 1024 or \
            trained["source_sha256"] != training["source_sha256"] or \
            trained["source_sha256"]["group_manifest"] != group_sha or \
            trained["source_sha256"]["expert_manifest"] != expert_sha or \
            trained["source_sha256"]["evidence_manifest"] != evidence_sha or \
            trained["source_sha256"]["frozen_weights"] != weights_sha or \
            trained["source_sha256"]["initial_checkpoint"] != initial_sha or \
            frozen["schema"] != "group4_joint_value_weights_v1" or \
            frozen["encoder_source_id"] != initial_sha or \
            audit["schema"] != "group4_prefix_contrast_dev_cache_audit_v1" or \
            audit["source_sha256"] != {"group_manifest": group_sha,
                                        "expert_manifest": expert_sha,
                                        "checkpoint": candidate_sha} or \
            audit["policy"]["trajectories"] != 160 or \
            audit["expert"]["prefix_contrasts"] != 303:
        raise ValueError("development provenance mismatch")
    for kind, manifest, root, shards in (
            ("policy", group, args.group_state_root, 1),
            ("expert", expert, args.expert_state_root, 3)):
        for shard in range(shards):
            path = root / "development" / f"summary_shard{shard}of{shards}.json"
            summary = json.loads(path.read_text())
            if summary["schema"] != "group4_prefix_contrast_dev_cache_shard_v1" or \
                    summary["kind"] != kind or summary["shard"] != shard or \
                    summary["shards"] != shards or \
                    summary["requested"] != len(
                        manifest["selected"]["development"][shard::shards]) or \
                    summary["completed"] != summary["requested"] or \
                    summary["source_id"] != candidate_sha:
                raise ValueError(f"development cache shard mismatch {kind}/{shard}")
    coverage, pairs = load_part(group, "development", args.group_turn_root,
                                args.group_state_root, group_sha, candidate_sha)
    expert_rows, expert_difference = expert_data(
        expert, "development", args.expert_state_root, expert_sha, candidate_sha)
    if coverage["matched_pairs"] != 103 or len(expert_rows) != 303:
        raise ValueError("development metric coverage changed")
    scale, vector = frozen["scale"].float(), frozen["vector"].float()
    with torch.no_grad():
        outcome_scores = F.normalize(torch.stack([p["difference"] for p in pairs]) /
                                     scale, dim=1) @ vector
        expert_scores = F.normalize(expert_difference / scale, dim=1) @ vector
    outcome = metrics(pairs, outcome_scores)
    instruction = expert_metrics(expert_rows, expert_scores)
    rows = evidence["selected"]["development"]
    baseline_rows = score_rows(args.initial_score_root, "initial", initial_sha,
                               evidence_sha, weights_sha, rows)
    candidate_rows = score_rows(args.candidate_score_root, "evidence", candidate_sha,
                                evidence_sha, weights_sha, rows)
    baseline, candidate = summarize_cross(baseline_rows), summarize_cross(candidate_rows)
    gate = {
        "group_outcome_accuracy_at_least_0_70": outcome["accuracy"] >= .70,
        "group_outcome_macro_at_least_0_70": outcome["group_macro_accuracy"] >= .70,
        "group_outcome_correct_at_least_72": outcome["correct"] >= 72,
        "cross_all_four_at_least_0_75": candidate["cross"]["all_four_accuracy"] >= .75,
        "cross_gain_over_initial_at_least_0_05":
            candidate["cross"]["all_four_accuracy"] -
            baseline["cross"]["all_four_accuracy"] >= .05,
        "onset_positive_at_least_0_65": candidate["onset"]["positive_rate"] >= .65,
        "onset_scene_macro_at_least_0_60":
            candidate["onset"]["scene_macro_positive_rate"] >= .60,
    }
    result = {
        "schema": "group4_evidence_onset_lora_development_v1",
        "interpretation": "reused train-scene development; no model audit, RL, or val-unseen result",
        "source_sha256": {
            "group_manifest": group_sha, "expert_manifest": expert_sha,
            "evidence_manifest": evidence_sha,
            "checkpoint": candidate_sha, "initial_checkpoint": initial_sha,
            "frozen_weights": weights_sha, "cache_audit": digest(args.cache_audit),
            "training_report": digest(args.training_report),
            "baseline_shards": [digest(args.initial_score_root / f"shard{i}.json")
                                for i in range(3)],
            "candidate_shards": [digest(args.candidate_score_root / f"shard{i}.json")
                                 for i in range(3)]},
        "coverage": coverage, "group_outcome": outcome,
        "broad_expert_instruction": instruction,
        "initial_evidence": baseline, "candidate_evidence": candidate,
        "predeclared_gate": gate,
        "eligible_for_policy_swap_development": all(gate.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
