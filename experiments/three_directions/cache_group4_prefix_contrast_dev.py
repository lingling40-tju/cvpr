"""Cache development states for a fixed, encoder-tuned group-four pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model
from fit_group4_future_success_linear import ANCHORS


def policy_id(row: dict) -> str:
    return f"s{row['seed']}_e{row['episode_id']}_v{row['variant']}"


def valid_tensor(value: torch.Tensor, shape: tuple[int, ...]) -> bool:
    return value.shape == shape and bool(torch.isfinite(value).all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("policy", "expert"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crossed-manifest", type=Path)
    parser.add_argument("--temporal-manifest", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("bad shard")
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = digest(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_id = digest(args.checkpoint)
    key = "group_manifest" if args.kind == "policy" else "expert_manifest"
    schema = ("policy_group_relative_manifest_v1" if args.kind == "policy"
              else "group4_joint_value_expert_manifest_v1")
    method = checkpoint["schema"]
    expected_steps = {"group4_prefix_contrast_lora_v1": 512,
                      "group4_crossed_prefix_lora_v1": 384,
                      "group4_temporal_interaction_lora_v1": 384,
                      "group4_evidence_onset_lora_v1": 384}
    if method == "group4_crossed_prefix_lora_v1" and (
            args.crossed_manifest is None or
            checkpoint["source_sha256"]["crossed_manifest"] !=
            digest(args.crossed_manifest)):
        raise ValueError("crossed manifest source mismatch")
    if method == "group4_temporal_interaction_lora_v1" and (
            args.temporal_manifest is None or
            checkpoint["source_sha256"]["temporal_manifest"] !=
            digest(args.temporal_manifest)):
        raise ValueError("temporal manifest source mismatch")
    if method == "group4_evidence_onset_lora_v1" and (
            args.evidence_manifest is None or
            checkpoint["source_sha256"]["evidence_manifest"] !=
            digest(args.evidence_manifest)):
        raise ValueError("evidence manifest source mismatch")
    if manifest["schema"] != schema or \
            method not in expected_steps or \
            checkpoint["source_sha256"][key] != manifest_sha or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
            checkpoint["training_microsteps"] != expected_steps[method]:
        raise ValueError("frozen adapted encoder provenance mismatch")
    rows = manifest["selected"]["development"][args.shard::args.shards]
    if not rows:
        raise ValueError("empty development cache shard")
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    output = args.output_root / "development"
    (output / "records").mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = reused = states = contrasts = 0
    for number, row in enumerate(rows, 1):
        if args.kind == "policy":
            rid = policy_id(row)
            record_path = args.record_root / "development" / "records" / f"{rid}.json"
            record = json.loads(record_path.read_text())
            if record["record_id"] != rid or \
                    record["manifest_sha256"] != manifest_sha or \
                    record["scene_id"] != row["scene_id"] or \
                    record["terminal_mode"] != row["terminal_mode"]:
                raise ValueError(f"bad policy record {rid}")
            last = max(t["original_turn_index"] for t in record["turns"])
            indices = [i for i, t in enumerate(record["turns"], 1)
                       if t["original_turn_index"] in ANCHORS and
                       t["original_turn_index"] < last]
            anchors = [record["turns"][i - 1]["original_turn_index"]
                       for i in indices]
            path = output / "records" / f"{rid}.pt"
            if path.is_file():
                prior = torch.load(path, map_location="cpu", weights_only=True)
                if prior["schema"] != "group_relative_state_cache_v1" or \
                        prior["record_id"] != rid or \
                        prior["source_id"] != source_id or \
                        prior["manifest_sha256"] != manifest_sha or \
                        prior["record_sha256"] != digest(record_path) or \
                        prior["indices"] != indices or \
                        prior["anchor_turns"] != anchors or \
                        not valid_tensor(prior["hidden"], (len(anchors), 2048)):
                    raise ValueError(f"bad reused policy state {rid}")
                completed += 1
                reused += 1
                states += len(anchors)
                continue
            item = {"record": record,
                    "root": args.record_root / "development"}
            values = []
            with torch.inference_mode():
                for index in indices:
                    inputs = build_inputs(processor, item, index).to("cuda")
                    hidden = model(**inputs, output_hidden_states=False,
                                   use_cache=False).logits[0, -1].float().cpu()
                    if not valid_tensor(hidden, (2048,)):
                        raise ValueError(f"nonfinite policy state {rid}/{index}")
                    values.append(hidden.to(torch.float16))
            payload = {"schema": "group_relative_state_cache_v1",
                       "record_id": rid, "source_id": source_id,
                       "manifest_sha256": manifest_sha,
                       "record_sha256": digest(record_path),
                       "indices": indices, "anchor_turns": anchors,
                       "hidden": torch.stack(values) if values else
                                 torch.empty((0, 2048), dtype=torch.float16)}
            states += len(anchors)
        else:
            eid = str(row["episode_id"])
            record_path = args.record_root / row["source_part"] / "records" / f"{eid}.json"
            if digest(record_path) != row["record_sha256"]:
                raise ValueError(f"expert source hash changed {eid}")
            record = json.loads(record_path.read_text())
            if record["episode_id"] != eid or \
                    record["scene_id"] != row["scene_id"] or \
                    record["trajectory_id"] != row["trajectory_id"] or \
                    record["turn_count"] != row["turn_count"] or \
                    record["wrong_instruction_episode_id"] != row["wrong_episode_id"]:
                raise ValueError(f"bad expert source {eid}")
            anchors = [a for a in ANCHORS if a < row["turn_count"]]
            path = output / "records" / f"{eid}.pt"
            if path.is_file():
                prior = torch.load(path, map_location="cpu", weights_only=True)
                if prior["schema"] != "group4_expert_prefix_state_v1" or \
                        prior["episode_id"] != eid or \
                        prior["part"] != "development" or \
                        prior["manifest_sha256"] != manifest_sha or \
                        prior["record_sha256"] != row["record_sha256"] or \
                        prior["source_id"] != source_id or \
                        prior["anchors"] != anchors or \
                        not valid_tensor(prior["correct"], (len(anchors), 2048)) or \
                        not valid_tensor(prior["wrong"], (len(anchors), 2048)):
                    raise ValueError(f"bad reused expert state {eid}")
                completed += 1
                reused += 1
                states += 2 * len(anchors)
                contrasts += len(anchors)
                continue
            item = {"record": record,
                    "root": args.record_root / row["source_part"]}
            correct, wrong = [], []
            with torch.inference_mode():
                for anchor in anchors:
                    for instruction, target in ((record["instruction"], correct),
                                                (record["wrong_instruction"], wrong)):
                        inputs = build_inputs(processor, item, anchor,
                                              instruction).to("cuda")
                        hidden = model(**inputs, output_hidden_states=False,
                                       use_cache=False).logits[0, -1].float().cpu()
                        if not valid_tensor(hidden, (2048,)):
                            raise ValueError(f"nonfinite expert state {eid}/{anchor}")
                        target.append(hidden.to(torch.float16))
            payload = {"schema": "group4_expert_prefix_state_v1",
                       "episode_id": eid, "part": "development",
                       "manifest_sha256": manifest_sha,
                       "record_sha256": row["record_sha256"],
                       "source_id": source_id, "anchors": anchors,
                       "correct": torch.stack(correct) if correct else
                                  torch.empty((0, 2048), dtype=torch.float16),
                       "wrong": torch.stack(wrong) if wrong else
                                torch.empty((0, 2048), dtype=torch.float16)}
            states += 2 * len(anchors)
            contrasts += len(anchors)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        completed += 1
        if number % 20 == 0:
            print(f"{args.kind} shard {args.shard}/{args.shards}: "
                  f"{number}/{len(rows)} elapsed_s={time.time()-started:.1f}",
                  flush=True)
    summary = {"schema": "group4_prefix_contrast_dev_cache_shard_v1",
               "kind": args.kind, "shard": args.shard,
               "shards": args.shards, "requested": len(rows),
               "completed": completed, "reused": reused,
               "states": states, "contrasts": contrasts,
               "manifest_sha256": manifest_sha, "source_id": source_id,
               "elapsed_seconds": time.time() - started}
    (output / f"summary_shard{args.shard}of{args.shards}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    if completed != len(rows):
        raise ValueError("incomplete development cache")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
