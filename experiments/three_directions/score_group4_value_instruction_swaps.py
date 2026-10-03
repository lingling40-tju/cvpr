"""Re-encode frozen audit histories with a natural wrong-goal instruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model
from fit_group4_future_success_linear import ANCHORS, FAILURES, SUCCESS


def group_shard(group_id: str, count: int) -> int:
    return int(hashlib.sha256(group_id.encode()).hexdigest(), 16) % count


def pair_margin(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor,
                vector: torch.Tensor) -> float:
    difference = F.normalize((a.float() - b.float()) / scale, dim=0)
    return float(difference @ vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--swaps", type=Path, required=True)
    parser.add_argument("--locked-audit", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--turn-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swapped-cache-root", type=Path)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("bad shard")
    manifest = json.loads(args.manifest.read_text())
    swaps = json.loads(args.swaps.read_text())
    locked = json.loads(args.locked_audit.read_text())
    state_audit = json.loads(args.state_audit.read_text())
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest_sha = digest(args.manifest)
    checkpoint_sha = digest(args.checkpoint)
    if manifest["schema"] != "policy_group_relative_manifest_v1" or \
            swaps["schema"] != "group4_future_success_instruction_swaps_v1" or \
            swaps["source_manifest_sha256"] != manifest_sha or \
            locked["schema"] != "group4_future_success_linear_locked_audit_v1" or \
            not locked["eligible_for_instruction_grounding_and_group4_wiring"] or \
            locked["weights_sha256"] != digest(args.weights) or \
            locked["manifest_sha256"] != manifest_sha or \
            state_audit["manifest_sha256"] != manifest_sha or \
            state_audit["source_id"] != checkpoint_sha or \
            weights["encoder_source_id"] != checkpoint_sha or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json"):
        raise ValueError("frozen swap/model provenance mismatch")
    expected_episodes = {str(row["episode_id"]) for row in
                         manifest["selected"]["audit"]}
    if set(swaps["swaps"]) != expected_episodes:
        raise ValueError("incomplete swapped-instruction coverage")
    plans = [plan for plan in manifest["selected"]["audit"]
             if group_shard(f"s{plan['seed']}_e{plan['episode_id']}", args.shards)
             == args.shard]
    if not plans:
        raise ValueError("empty shard")
    scale, vector = weights["scale"].float(), weights["vector"].float()
    if scale.shape != (2048,) or vector.shape != (2048,):
        raise ValueError("invalid frozen value head")
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    groups = {}
    with torch.inference_mode():
        for number, plan in enumerate(plans, 1):
            eid = str(plan["episode_id"])
            gid = f"s{plan['seed']}_e{eid}"
            rid = f"{gid}_v{plan['variant']}"
            swap = swaps["swaps"][eid]
            record_path = args.turn_root / "audit" / "records" / f"{rid}.json"
            record = json.loads(record_path.read_text())
            cache = torch.load(args.state_root / "audit" / "records" /
                               f"{rid}.pt", map_location="cpu", weights_only=True)
            if record["record_id"] != rid or record["manifest_sha256"] != manifest_sha or \
                    record["scene_id"] != plan["scene_id"] or \
                    record["instruction"].strip() != swap["original_instruction"] or \
                    cache["record_id"] != rid or cache["source_id"] != checkpoint_sha or \
                    cache["manifest_sha256"] != manifest_sha:
                raise ValueError(f"invalid audit history {rid}")
            last = max(turn["original_turn_index"] for turn in record["turns"])
            item = {"record": record, "root": args.turn_root / "audit"}
            features = {}
            positions = [(count, anchor, original) for count, anchor, original in
                         zip(cache["indices"], cache["anchor_turns"], cache["hidden"])
                         if anchor in ANCHORS and anchor < last]
            cached_vectors = None
            cached_path = None
            if args.swapped_cache_root:
                cached_path = args.swapped_cache_root / "records" / f"{rid}.pt"
                if cached_path.is_file():
                    prior = torch.load(cached_path, map_location="cpu",
                                       weights_only=True)
                    if prior["schema"] != "group4_policy_wrong_instruction_state_v1" or \
                            prior["record_id"] != rid or \
                            prior["manifest_sha256"] != manifest_sha or \
                            prior["swaps_sha256"] != digest(args.swaps) or \
                            prior["source_id"] != checkpoint_sha or \
                            prior["record_sha256"] != digest(record_path) or \
                            prior["anchors"] != [p[1] for p in positions] or \
                            prior["hidden"].shape != (len(positions), 2048) or \
                            not bool(torch.isfinite(prior["hidden"]).all()):
                        raise ValueError(f"invalid reusable swapped state {rid}")
                    cached_vectors = prior["hidden"].float()
            if cached_vectors is None:
                vectors = []
                for count, anchor, _ in positions:
                    inputs = build_inputs(processor, item, count,
                                          swap["swapped_instruction"]).to("cuda")
                    output = model(**inputs, output_hidden_states=False,
                                   use_cache=False)
                    swapped = output.logits[0, -1].float().cpu()
                    if not bool(torch.isfinite(swapped).all()):
                        raise ValueError(f"nonfinite swapped state {rid}/{anchor}")
                    vectors.append(swapped)
                cached_vectors = torch.stack(vectors).float() if vectors else \
                    torch.empty((0, 2048), dtype=torch.float32)
                if cached_path is not None:
                    cached_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "schema": "group4_policy_wrong_instruction_state_v1",
                        "record_id": rid, "manifest_sha256": manifest_sha,
                        "swaps_sha256": digest(args.swaps),
                        "source_id": checkpoint_sha,
                        "record_sha256": digest(record_path),
                        "anchors": [p[1] for p in positions],
                        "hidden": cached_vectors.float()}
                    temporary = cached_path.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    os.replace(temporary, cached_path)
            for (_, anchor, original), swapped in zip(positions, cached_vectors):
                features[anchor] = (original.float(), swapped.float())
            group = groups.setdefault(gid, {"group_id": gid, "episode_id": eid,
                                            "scene_id": plan["scene_id"],
                                            "same_start_swap":
                                                swap["start_separation_m_for_audit_only"] <= .5,
                                            "variants": {}})
            if plan["variant"] in group["variants"]:
                raise ValueError(f"repeated variant {rid}")
            group["variants"][plan["variant"]] = {
                "terminal_mode": plan["terminal_mode"], "features": features}
            if number % 10 == 0:
                print(f"swap shard {args.shard}: {number}/{len(plans)}", flush=True)
    output_groups = []
    for group in groups.values():
        if len(group["variants"]) != 4:
            raise ValueError(f"incomplete group {group['group_id']}")
        pairs = []
        for anchor in ANCHORS:
            for good_variant, good in group["variants"].items():
                if good["terminal_mode"] != SUCCESS or anchor not in good["features"]:
                    continue
                for bad_variant, bad in group["variants"].items():
                    if bad["terminal_mode"] not in FAILURES or anchor not in bad["features"]:
                        continue
                    g0, g1 = good["features"][anchor]
                    b0, b1 = bad["features"][anchor]
                    pairs.append({"anchor": anchor,
                                  "preferred_variant": good_variant,
                                  "rejected_variant": bad_variant,
                                  "original_margin": pair_margin(g0, b0, scale, vector),
                                  "swapped_margin": pair_margin(g1, b1, scale, vector)})
        output_groups.append({key: value for key, value in group.items()
                              if key != "variants"} | {"pairs": pairs})
    result = {"schema": "group4_value_instruction_swap_shard_v1",
              "shard": args.shard, "shards": args.shards,
              "manifest_sha256": manifest_sha,
              "swaps_sha256": digest(args.swaps),
              "locked_audit_sha256": digest(args.locked_audit),
              "state_audit_sha256": digest(args.state_audit),
              "weights_sha256": digest(args.weights),
              "checkpoint_sha256": checkpoint_sha,
              "trajectory_count": len(plans),
              "group_count": len(output_groups),
              "pair_count": sum(len(group["pairs"]) for group in output_groups),
              "groups": sorted(output_groups, key=lambda row: row["group_id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"shard": args.shard,
                      "trajectory_count": len(plans),
                      "group_count": len(output_groups),
                      "pair_count": result["pair_count"]}, indent=2))


if __name__ == "__main__":
    main()
