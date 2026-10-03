"""Score pre-frozen evidence-conditioned development branches in shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model


def margin(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor,
           vector: torch.Tensor) -> float:
    return float(F.normalize((a.float() - b.float()) / scale, dim=0) @ vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("initial", "evidence"), required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    manifest_sha = digest(args.manifest)
    checkpoint_sha = digest(args.checkpoint)
    if manifest["schema"] != "group4_evidence_onset_manifest_v1" or \
            manifest["inventory"]["development"]["crossed_pairs"] != 94 or \
            manifest["inventory"]["development"]["evidence_onset_eligible"] != 44 or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
            weights["schema"] != "group4_joint_value_weights_v1":
        raise ValueError("development source mismatch")
    if args.source_kind == "initial":
        if checkpoint["schema"] != "history_grounding_lora_interim_v1" or \
                checkpoint_sha != weights["encoder_source_id"]:
            raise ValueError("wrong original SFT encoder")
    elif checkpoint["schema"] != "group4_evidence_onset_lora_v1" or \
            checkpoint["source_sha256"]["evidence_manifest"] != manifest_sha or \
            checkpoint["source_sha256"]["frozen_weights"] != digest(args.frozen_weights):
        raise ValueError("wrong evidence-onset encoder")
    rows = manifest["selected"]["development"][args.shard::args.shards]
    if not rows:
        raise ValueError("empty development shard")
    scale, vector = weights["scale"].float(), weights["vector"].float()
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    result_rows = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            a, b, anchor = row["episode_a"], row["episode_b"], row["anchor"]
            root_a = args.expert_root / row["source_part_a"]
            root_b = args.expert_root / row["source_part_b"]
            pa = root_a / "records" / f"{a}.json"
            pb = root_b / "records" / f"{b}.json"
            if digest(pa) != row["record_a_sha256"] or \
                    digest(pb) != row["record_b_sha256"]:
                raise ValueError(f"expert record changed {a}/{b}")
            ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
            if ra["scene_id"] != row["scene_id"] or \
                    rb["scene_id"] != row["scene_id"] or \
                    ra["turn_count"] <= anchor or rb["turn_count"] <= anchor:
                raise ValueError(f"expert pair changed {a}/{b}")
            ia, ib = ra["instruction"], rb["instruction"]
            item_a, item_b = {"record": ra, "root": root_a}, \
                             {"record": rb, "root": root_b}

            def encode(item: dict, count: int, instruction: str) -> torch.Tensor:
                inputs = build_inputs(processor, item, count, instruction).to("cuda")
                hidden = model(**inputs, output_hidden_states=False,
                               use_cache=False).logits[0, -1].float().cpu()
                if hidden.shape != (2048,) or not bool(torch.isfinite(hidden).all()):
                    raise ValueError(f"nonfinite state {a}/{b}")
                return hidden.to(torch.float16)

            aa, ab = encode(item_a, anchor, ia), encode(item_a, anchor, ib)
            bb, ba = encode(item_b, anchor, ib), encode(item_b, anchor, ia)
            measured = {"episode_a": a, "episode_b": b,
                        "scene_id": row["scene_id"], "anchor": anchor,
                        "evidence_onset_eligible": row["evidence_onset_eligible"],
                        "row_a": margin(aa, ab, scale, vector),
                        "row_b": margin(bb, ba, scale, vector),
                        "column_a": margin(aa, ba, scale, vector),
                        "column_b": margin(bb, ab, scale, vector)}
            if row["evidence_onset_eligible"]:
                a3a, a3b = encode(item_a, 3, ia), encode(item_a, 3, ib)
                b3b, b3a = encode(item_b, 3, ib), encode(item_b, 3, ia)
                measured["onset_delta_a"] = measured["row_a"] - margin(
                    a3a, a3b, scale, vector)
                measured["onset_delta_b"] = measured["row_b"] - margin(
                    b3b, b3a, scale, vector)
            result_rows.append(measured)
            if number % 10 == 0:
                print(f"{args.source_kind} shard {args.shard}: "
                      f"{number}/{len(rows)}", flush=True)
    result = {"schema": "group4_evidence_onset_dev_score_shard_v1",
              "source_kind": args.source_kind, "shard": args.shard,
              "shards": args.shards, "manifest_sha256": manifest_sha,
              "checkpoint_sha256": checkpoint_sha,
              "frozen_weights_sha256": digest(args.frozen_weights),
              "pairs": len(result_rows), "rows": result_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
