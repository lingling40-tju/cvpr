"""Audit cached wrong-goal policy states and score a frozen joint readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_group4_future_success_linear import ANCHORS, FAILURES, SUCCESS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin(good: torch.Tensor, bad: torch.Tensor, scale: torch.Tensor,
           vector: torch.Tensor) -> float:
    return float(F.normalize((good.float() - bad.float()) / scale, dim=0)
                 @ vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--swaps", type=Path, required=True)
    parser.add_argument("--locked-audit", type=Path, required=True)
    parser.add_argument("--group-state-audit", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--swapped-cache-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    swaps = json.loads(args.swaps.read_text())
    locked = json.loads(args.locked_audit.read_text())
    state_audit = json.loads(args.group_state_audit.read_text())
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    manifest_sha = digest(args.manifest)
    swaps_sha = digest(args.swaps)
    source_id = digest(args.checkpoint)
    if manifest["schema"] != "policy_group_relative_manifest_v1" or \
            swaps["schema"] != "group4_future_success_instruction_swaps_v1" or \
            swaps["source_manifest_sha256"] != manifest_sha or \
            locked["schema"] != "group4_joint_value_locked_audit_v1" or \
            not locked["eligible_for_policy_instruction_recheck"] or \
            locked["source_sha256"]["group_manifest"] != manifest_sha or \
            locked["source_sha256"]["weights"] != digest(args.weights) or \
            weights["schema"] != "group4_joint_value_weights_v1" or \
            weights["encoder_source_id"] != source_id or \
            state_audit["manifest_sha256"] != manifest_sha or \
            state_audit["source_id"] != source_id:
        raise ValueError("frozen source/model mismatch")
    rows = manifest["selected"]["audit"]
    if len(rows) != 160 or len({f"s{r['seed']}_e{r['episode_id']}" for r in rows}) != 40 or \
            set(swaps["swaps"]) != {str(r["episode_id"]) for r in rows}:
        raise ValueError("unexpected policy audit coverage")
    scale, vector = weights["scale"].float(), weights["vector"].float()
    if scale.shape != (2048,) or vector.shape != (2048,) or \
            not bool(torch.isfinite(scale).all()) or \
            not bool(torch.isfinite(vector).all()):
        raise ValueError("invalid joint weights")
    groups: dict[str, dict] = {}
    states = 0
    for plan in rows:
        eid = str(plan["episode_id"])
        gid = f"s{plan['seed']}_e{eid}"
        rid = f"{gid}_v{plan['variant']}"
        record_path = args.turn_root / "audit" / "records" / f"{rid}.json"
        original_path = args.state_root / "audit" / "records" / f"{rid}.pt"
        swapped_path = args.swapped_cache_root / "records" / f"{rid}.pt"
        record = json.loads(record_path.read_text())
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        wrong = torch.load(swapped_path, map_location="cpu", weights_only=True)
        swap = swaps["swaps"][eid]
        last = max(t["original_turn_index"] for t in record["turns"])
        positions = [(int(anchor), vec.float()) for anchor, vec in
                     zip(original["anchor_turns"], original["hidden"])
                     if anchor in ANCHORS and anchor < last]
        anchors = [anchor for anchor, _ in positions]
        if record["record_id"] != rid or \
                record["manifest_sha256"] != manifest_sha or \
                record["scene_id"] != plan["scene_id"] or \
                record["instruction"].strip() != swap["original_instruction"] or \
                original["record_id"] != rid or \
                original["manifest_sha256"] != manifest_sha or \
                original["source_id"] != source_id or \
                wrong["schema"] != "group4_policy_wrong_instruction_state_v1" or \
                wrong["record_id"] != rid or \
                wrong["manifest_sha256"] != manifest_sha or \
                wrong["swaps_sha256"] != swaps_sha or \
                wrong["source_id"] != source_id or \
                wrong["record_sha256"] != digest(record_path) or \
                wrong["anchors"] != anchors or \
                wrong["hidden"].shape != (len(anchors), 2048) or \
                not bool(torch.isfinite(wrong["hidden"]).all()):
            raise ValueError(f"bad policy swapped cache {rid}")
        group = groups.setdefault(gid, {"episode_id": eid,
                                        "scene_id": plan["scene_id"],
                                        "variants": {}})
        if group["scene_id"] != plan["scene_id"] or \
                plan["variant"] in group["variants"]:
            raise ValueError(f"bad group membership {gid}")
        group["variants"][plan["variant"]] = {
            "terminal_mode": plan["terminal_mode"],
            "original": dict(positions),
            "wrong": dict(zip(anchors, wrong["hidden"].float()))}
        states += len(anchors)
    if len(groups) != 40 or any(len(g["variants"]) != 4 for g in groups.values()) or \
            states != locked["coverage"]["group"]["states"]:
        raise ValueError("incomplete cached state coverage")
    active = []
    for gid, group in sorted(groups.items()):
        pairs = []
        for anchor in ANCHORS:
            for gv, good in group["variants"].items():
                if good["terminal_mode"] != SUCCESS or anchor not in good["original"]:
                    continue
                for bv, bad in group["variants"].items():
                    if bad["terminal_mode"] not in FAILURES or anchor not in bad["original"]:
                        continue
                    pairs.append({"anchor": anchor, "preferred_variant": gv,
                                  "rejected_variant": bv,
                                  "original_margin": margin(
                                      good["original"][anchor],
                                      bad["original"][anchor], scale, vector),
                                  "swapped_margin": margin(
                                      good["wrong"][anchor],
                                      bad["wrong"][anchor], scale, vector)})
        if pairs:
            active.append({"group_id": gid, "scene_id": group["scene_id"],
                           "same_start_swap": swaps["swaps"][group["episode_id"]]
                           ["start_separation_m_for_audit_only"] <= .5,
                           "pairs": pairs})
    pairs = [p for g in active for p in g["pairs"]]
    original_correct = sum(p["original_margin"] > 0 for p in pairs)
    swapped_correct = sum(p["swapped_margin"] > 0 for p in pairs)
    original_macro = sum(sum(p["original_margin"] > 0 for p in g["pairs"])
                         / len(g["pairs"]) for g in active) / len(active)
    if len(active) != locked["outcome"]["groups"] or \
            len(pairs) != locked["outcome"]["pairs"] or \
            original_correct != locked["outcome"]["correct"] or \
            abs(original_macro - locked["outcome"]["group_macro_accuracy"]) > 1e-9:
        raise ValueError("correct-instruction scores differ from locked audit")
    group_margin_drop = sum(
        sum(p["original_margin"] - p["swapped_margin"] for p in g["pairs"])
        / len(g["pairs"]) > 0 for g in active)
    rng = random.Random(11)
    clustered = []
    for _ in range(5000):
        sampled = [rng.choice(active) for _ in active]
        n = sum(len(g["pairs"]) for g in sampled)
        clustered.append(sum(sum((p["original_margin"] > 0) -
                                 (p["swapped_margin"] > 0) for p in g["pairs"])
                             for g in sampled) / n)
    clustered.sort()
    gate = {"all_131_pairs_covered": len(pairs) == 131,
            "all_22_mixed_groups_covered": len(active) == 22,
            "swapped_accuracy_drop_at_least_0_10":
                (original_correct - swapped_correct) / len(pairs) >= .10,
            "group_margin_drop_fraction_at_least_0_60":
                group_margin_drop / len(active) >= .60}
    result = {"schema": "group4_joint_policy_instruction_swap_analysis_v1",
              "source_sha256": {"manifest": manifest_sha, "swaps": swaps_sha,
                                "locked_audit": digest(args.locked_audit),
                                "group_state_audit": digest(args.group_state_audit),
                                "weights": digest(args.weights),
                                "encoder_checkpoint": source_id},
              "cache_coverage": {"trajectories": len(rows), "groups": len(groups),
                                 "preterminal_states": states,
                                 "matched_groups": len(active),
                                 "matched_pairs": len(pairs)},
              "original_correct": original_correct,
              "original_accuracy": original_correct / len(pairs),
              "swapped_correct": swapped_correct,
              "swapped_accuracy": swapped_correct / len(pairs),
              "accuracy_drop": (original_correct - swapped_correct) / len(pairs),
              "group_margin_drop": group_margin_drop,
              "group_margin_drop_fraction": group_margin_drop / len(active),
              "group_cluster_bootstrap_accuracy_drop_95":
                  [clustered[125], clustered[4874]],
              "same_start_active_groups": sum(g["same_start_swap"] for g in active),
              "same_start_pairs": sum(len(g["pairs"]) for g in active
                                      if g["same_start_swap"]),
              "predeclared_gate": gate,
              "eligible_for_group4_reward_wiring": all(gate.values()),
              "interpretation": "natural wrong-goal policy-history diagnostic; not independent semantic or navigation validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
