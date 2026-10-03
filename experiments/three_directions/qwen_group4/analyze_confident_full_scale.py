"""Audit three matched full-val confidence runs and unused screen episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_matched_pair import load_label
from analyze_qwen_group_full_scale import digest, SEEDS, summarize_scope


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "e9b67757d2f92384deefa6f619633d0c99450357f762a1775f5099eb6a16f7c1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full_path = args.root / "manifest.json"
    if digest(full_path) != FULL_SHA or digest(args.screen_manifest) != SCREEN_SHA:
        raise ValueError("fixed full/screen manifest hashes changed")
    full = json.loads(full_path.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = [str(x) for x in full["episode_ids"]]
    scenes = [str(x) for x in full["scene_ids"]]
    screen_ids = {str(x) for x in screen["episode_ids"]}
    if len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != 1839 or \
            len(screen_ids) != 256 or not screen_ids.issubset(ids):
        raise ValueError("invalid full/screen episode coverage")
    outside_ids = [eid for eid in ids if eid not in screen_ids]
    outside_scenes = [scene for eid, scene in zip(ids, scenes)
                      if eid not in screen_ids]
    if len(outside_ids) != 1583:
        raise ValueError("screen complement should contain 1,583 episodes")
    rows_by_seed = {}
    pair_hashes = {}
    for seed in SEEDS:
        candidate = f"qwen_confident_scale_128_seed{seed}"
        control = f"qwen_confident_scale_control_128_seed{seed}"
        pair_path = args.root / f"paired_{candidate}_vs_{control}.json"
        pair = json.loads(pair_path.read_text())
        if pair["split"] != "val_unseen" or pair["episodes"] != 1839 or \
                pair["manifest_sha256"] != FULL_SHA or \
                pair["candidate"] != candidate or pair["control"] != control or \
                pair["candidate_metrics"]["inference_errors"] != 0 or \
                pair["control_metrics"]["inference_errors"] != 0:
            raise ValueError(f"invalid paired full evaluation seed {seed}")
        rows_by_seed[seed] = (load_label(args.root, candidate, ids, 4),
                              load_label(args.root, control, ids, 4))
        pair_hashes[str(seed)] = digest(pair_path)
    report = {"schema": "qwen_confident_full_val_unseen_v1",
              "full_manifest_sha256": FULL_SHA,
              "screen_manifest_sha256": SCREEN_SHA,
              "paired_json_sha256": pair_hashes,
              "seeds": list(SEEDS),
              "full": summarize_scope(rows_by_seed, ids, scenes),
              "outside_reused_screen": summarize_scope(
                  rows_by_seed, outside_ids, outside_scenes),
              "interpretation": "Three paired seeds, one decode per checkpoint/episode. Scene-and-seed bootstrap is exploratory; no independent test split or human semantic truth."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({scope: report[scope]["metrics"]
                      for scope in ("full", "outside_reused_screen")}, indent=2))


if __name__ == "__main__":
    main()
