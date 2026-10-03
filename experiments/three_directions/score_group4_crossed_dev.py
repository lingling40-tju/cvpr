"""Score frozen 2x2 same-start trajectory/instruction crosses on development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import set_peft_model_state_dict

from history_grounding_lora import build_inputs, digest, load_model


def score(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor,
          vector: torch.Tensor) -> float:
    return float(F.normalize((a.float() - b.float()) / scale, dim=0)
                 @ vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossed-manifest", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("initial", "adapted"), required=True)
    parser.add_argument("--frozen-weights", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("bad shard")
    crossed = json.loads(args.crossed_manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = torch.load(args.frozen_weights, map_location="cpu", weights_only=True)
    crossed_sha = digest(args.crossed_manifest)
    checkpoint_sha = digest(args.checkpoint)
    if crossed["schema"] != "group4_crossed_prefix_manifest_v1" or \
            crossed["inventory"]["development"]["crossed_pairs"] != 56 or \
            checkpoint["model_config_sha256"] != digest(args.model / "config.json") or \
            weights["schema"] != "group4_joint_value_weights_v1":
        raise ValueError("crossed development source mismatch")
    if args.source_kind == "initial":
        if checkpoint["schema"] != "history_grounding_lora_interim_v1" or \
                checkpoint_sha != weights["encoder_source_id"]:
            raise ValueError("wrong initial encoder")
    elif checkpoint["schema"] != "group4_crossed_prefix_lora_v1" or \
            checkpoint["source_sha256"]["crossed_manifest"] != crossed_sha or \
            checkpoint["source_sha256"]["frozen_weights"] != \
            digest(args.frozen_weights):
        raise ValueError("wrong crossed encoder")
    rows = crossed["selected"]["development"][args.shard::args.shards]
    if not rows:
        raise ValueError("empty crossed score shard")
    scale, vector = weights["scale"].float(), weights["vector"].float()
    processor, model, _, _ = load_model(args.model)
    set_peft_model_state_dict(model, checkpoint["adapter"])
    model.eval()
    result_rows = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            a, b = row["episode_a"], row["episode_b"]
            root_a = args.expert_root / row["source_part_a"]
            root_b = args.expert_root / row["source_part_b"]
            pa, pb = root_a / "records" / f"{a}.json", \
                     root_b / "records" / f"{b}.json"
            if digest(pa) != row["record_a_sha256"] or \
                    digest(pb) != row["record_b_sha256"]:
                raise ValueError(f"expert record changed {a}/{b}")
            ra, rb = json.loads(pa.read_text()), json.loads(pb.read_text())
            if ra["scene_id"] != row["scene_id"] or \
                    rb["scene_id"] != row["scene_id"] or \
                    ra["turn_count"] <= 6 or rb["turn_count"] <= 6:
                raise ValueError(f"expert cross changed {a}/{b}")
            ia, ib = ra["instruction"], rb["instruction"]
            item_a = {"record": ra, "root": root_a}
            item_b = {"record": rb, "root": root_b}
            def encode(item: dict, instruction: str) -> torch.Tensor:
                inputs = build_inputs(processor, item, 6, instruction).to("cuda")
                hidden = model(**inputs, output_hidden_states=False,
                               use_cache=False).logits[0, -1].float().cpu()
                if hidden.shape != (2048,) or \
                        not bool(torch.isfinite(hidden).all()):
                    raise ValueError(f"nonfinite crossed dev state {a}/{b}")
                return hidden.to(torch.float16)
            aa = encode(item_a, ia)
            ab = encode(item_a, ib)
            bb = encode(item_b, ib)
            ba = encode(item_b, ia)
            result_rows.append({"episode_a": a, "episode_b": b,
                                "scene_id": row["scene_id"],
                                "row_a": score(aa, ab, scale, vector),
                                "row_b": score(bb, ba, scale, vector),
                                "column_a": score(aa, ba, scale, vector),
                                "column_b": score(bb, ab, scale, vector)})
            if number % 10 == 0:
                print(f"{args.source_kind} crossed shard {args.shard}: "
                      f"{number}/{len(rows)}", flush=True)
    result = {"schema": "group4_crossed_dev_score_shard_v1",
              "source_kind": args.source_kind,
              "shard": args.shard, "shards": args.shards,
              "crossed_manifest_sha256": crossed_sha,
              "checkpoint_sha256": checkpoint_sha,
              "frozen_weights_sha256": digest(args.frozen_weights),
              "pairs": len(result_rows), "rows": result_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
