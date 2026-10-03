"""Evaluate goal-conditioned temporal value change on development states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import digest, load_part, metrics
from fit_group4_expert_prefix_value import expert_data, expert_metrics


def temporal_data(manifest: dict, state_root: Path, source_id: str,
                  expert_sha: str) -> tuple[list[dict], torch.Tensor]:
    rows = manifest["selected"]["development"]
    differences = []
    for row in rows:
        eid = row["episode_id"]
        cache = torch.load(state_root / "development" / "records" / f"{eid}.pt",
                           map_location="cpu", weights_only=True)
        if cache["schema"] != "group4_expert_prefix_state_v1" or \
                cache["episode_id"] != eid or \
                cache["manifest_sha256"] != expert_sha or \
                cache["record_sha256"] != row["record_sha256"] or \
                cache["source_id"] != source_id or \
                3 not in cache["anchors"] or 6 not in cache["anchors"]:
            raise ValueError(f"bad temporal cache {eid}")
        i3, i6 = cache["anchors"].index(3), cache["anchors"].index(6)
        c3, c6 = cache["correct"][i3].float(), cache["correct"][i6].float()
        w3, w6 = cache["wrong"][i3].float(), cache["wrong"][i6].float()
        difference = (c6 - c3) - (w6 - w3)
        if difference.shape != (2048,) or \
                not bool(torch.isfinite(difference).all()):
            raise ValueError(f"nonfinite temporal difference {eid}")
        differences.append(difference)
    return rows, torch.stack(differences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--group-turn-root", type=Path, required=True)
    parser.add_argument("--group-state-root", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--temporal-manifest", type=Path, required=True)
    parser.add_argument("--expert-state-root", type=Path, required=True)
    parser.add_argument("--initial-expert-state-root", type=Path, required=True)
    parser.add_argument("--prefix-expert-state-root", type=Path, required=True)
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--crossed-expert-state-root", type=Path, required=True)
    parser.add_argument("--crossed-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--previous-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.group_manifest.read_text())
    expert = json.loads(args.expert_manifest.read_text())
    temporal = json.loads(args.temporal_manifest.read_text())
    trained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    report = json.loads(args.training_report.read_text())
    cache_audit = json.loads(args.cache_audit.read_text())
    frozen = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    previous = json.loads(args.previous_development.read_text())
    group_sha = digest(args.group_manifest)
    expert_sha = digest(args.expert_manifest)
    temporal_sha = digest(args.temporal_manifest)
    source_id = digest(args.checkpoint)
    if group["schema"] != "policy_group_relative_manifest_v1" or \
            expert["schema"] != "group4_joint_value_expert_manifest_v1" or \
            temporal["schema"] != "group4_temporal_interaction_manifest_v1" or \
            temporal["source_expert_manifest_sha256"] != expert_sha or \
            temporal["inventory"]["development"]["temporal_interactions"] != 143 or \
            trained["schema"] != "group4_temporal_interaction_lora_v1" or \
            trained["source_sha256"] != report["source_sha256"] or \
            report["schema"] != "group4_temporal_interaction_lora_training_v1" or \
            report["smoke"] or report["microsteps"] != 384 or \
            report["model_forward_count"] != 1024 or \
            trained["source_sha256"]["group_manifest"] != group_sha or \
            trained["source_sha256"]["expert_manifest"] != expert_sha or \
            trained["source_sha256"]["temporal_manifest"] != temporal_sha or \
            trained["source_sha256"]["frozen_weights"] != digest(args.frozen_weights) or \
            frozen["schema"] != "group4_joint_value_weights_v1" or \
            previous["schema"] != "group4_crossed_prefix_lora_development_v1":
        raise ValueError("frozen development source mismatch")
    if cache_audit["schema"] != "group4_prefix_contrast_dev_cache_audit_v1" or \
            cache_audit["source_sha256"] != {
                "group_manifest": group_sha,
                "expert_manifest": expert_sha,
                "checkpoint": source_id} or \
            cache_audit["policy"]["trajectories"] != 160 or \
            cache_audit["expert"]["prefix_contrasts"] != 303:
        raise ValueError("independent development cache audit failed")
    for kind, manifest, root, shards in (
            ("policy", group, args.group_state_root, 1),
            ("expert", expert, args.expert_state_root, 3)):
        rows = manifest["selected"]["development"]
        for shard in range(shards):
            summary = json.loads((root / "development" /
                                  f"summary_shard{shard}of{shards}.json").read_text())
            if summary["schema"] != "group4_prefix_contrast_dev_cache_shard_v1" or \
                    summary["kind"] != kind or summary["shard"] != shard or \
                    summary["shards"] != shards or \
                    summary["requested"] != len(rows[shard::shards]) or \
                    summary["completed"] != len(rows[shard::shards]) or \
                    summary["manifest_sha256"] != digest(
                        args.group_manifest if kind == "policy" else args.expert_manifest) or \
                    summary["source_id"] != source_id:
                raise ValueError(f"incomplete development shard {kind}/{shard}")
    coverage, group_pairs = load_part(
        group, "development", args.group_turn_root,
        args.group_state_root, group_sha, source_id)
    expert_rows, expert_difference = expert_data(
        expert, "development", args.expert_state_root, expert_sha, source_id)
    if coverage["trajectories"] != 160 or \
            coverage["matched_pairs"] != 103 or \
            coverage["matched_groups"] != 18 or \
            len(expert_rows) != 303 or \
            len({row["scene_id"] for row in expert_rows}) != 8:
        raise ValueError("development metric coverage mismatch")
    scale = frozen["scale"].float()
    vector = frozen["vector"].float()
    if scale.shape != (2048,) or vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all()) or \
            not bool(torch.isfinite(vector).all()):
        raise ValueError("bad frozen readout")
    with torch.no_grad():
        outcome_x = F.normalize(
            torch.stack([row["difference"] for row in group_pairs]) / scale,
            dim=1)
        expert_x = F.normalize(expert_difference / scale, dim=1)
        outcome_scores = outcome_x @ vector
        expert_scores = expert_x @ vector
    outcome = metrics(group_pairs, outcome_scores)
    instruction = expert_metrics(expert_rows, expert_scores)
    temporal_rows, temporal_difference = temporal_data(
        temporal, args.expert_state_root, source_id, expert_sha)
    old_temporal_rows, old_temporal_difference = temporal_data(
        temporal, args.initial_expert_state_root,
        frozen["encoder_source_id"], expert_sha)
    prefix_sha = digest(args.prefix_checkpoint)
    crossed_sha = digest(args.crossed_checkpoint)
    prefix_rows, prefix_difference = temporal_data(
        temporal, args.prefix_expert_state_root, prefix_sha, expert_sha)
    crossed_rows, crossed_difference = temporal_data(
        temporal, args.crossed_expert_state_root, crossed_sha, expert_sha)
    if len(temporal_rows) != 143 or temporal_rows != old_temporal_rows:
        raise ValueError("temporal development coverage changed")
    if temporal_rows != prefix_rows or temporal_rows != crossed_rows:
        raise ValueError("temporal comparison not paired")
    with torch.no_grad():
        temporal_scores = F.normalize(temporal_difference / scale, dim=1) @ vector
        old_temporal_scores = F.normalize(old_temporal_difference / scale,
                                           dim=1) @ vector
        prefix_scores = F.normalize(prefix_difference / scale, dim=1) @ vector
        crossed_scores = F.normalize(crossed_difference / scale, dim=1) @ vector
    temporal_result = expert_metrics(temporal_rows, temporal_scores)
    old_temporal = expert_metrics(temporal_rows, old_temporal_scores)
    prefix_temporal = expert_metrics(temporal_rows, prefix_scores)
    crossed_temporal = expert_metrics(temporal_rows, crossed_scores)
    old = previous["previous_frozen_joint"]
    if old["outcome"]["correct"] != 74 or old["expert"]["correct"] != 196 or \
            previous["outcome"]["correct"] != 78 or \
            previous["expert_prefix_instruction"]["correct"] != 215 or \
            old_temporal["correct"] != 97 or \
            prefix_temporal["correct"] != 99 or \
            crossed_temporal["correct"] != 104:
        raise ValueError("old development control changed")
    gate = {"outcome_accuracy_at_least_0_70": outcome["accuracy"] >= .70,
            "outcome_group_macro_at_least_0_70":
                outcome["group_macro_accuracy"] >= .70,
            "outcome_no_more_than_two_pairs_below_old_joint":
                outcome["correct"] >= old["outcome"]["correct"] - 2,
            "expert_prefix_accuracy_at_least_0_75":
                instruction["accuracy"] >= .75,
            "expert_prefix_scene_macro_at_least_0_70":
                instruction["scene_macro_accuracy"] >= .70,
            "temporal_interaction_accuracy_at_least_0_75":
                temporal_result["accuracy"] >= .75,
            "temporal_interaction_scene_macro_at_least_0_70":
                temporal_result["scene_macro_accuracy"] >= .70}
    result = {"schema": "group4_temporal_interaction_lora_development_v1",
              "interpretation": "scene-disjoint development of fixed encoder pilot; no navigation or blind-audit result",
              "source_sha256": {"group_manifest": group_sha,
                                "expert_manifest": expert_sha,
                                "temporal_manifest": temporal_sha,
                                "checkpoint": source_id,
                                "prefix_checkpoint": prefix_sha,
                                "crossed_checkpoint": crossed_sha,
                                "cache_audit": digest(args.cache_audit),
                                "training_report": digest(args.training_report),
                                "frozen_weights": digest(args.frozen_weights),
                                "previous_development": digest(args.previous_development)},
              "coverage": {"group": coverage,
                           "expert_prefix_pairs": len(expert_rows)},
              "outcome": outcome, "expert_prefix_instruction": instruction,
              "temporal_interaction": temporal_result,
              "initial_temporal_interaction": old_temporal,
              "prefix_pilot_temporal_interaction": prefix_temporal,
              "crossed_pilot_temporal_interaction": crossed_temporal,
              "previous_frozen_joint": old,
              "previous_crossed_encoder_pilot": {
                  "outcome": previous["outcome"],
                  "expert_prefix_instruction":
                      previous["expert_prefix_instruction"]},
              "predeclared_gate": gate,
              "eligible_for_policy_swap_development": all(gate.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
