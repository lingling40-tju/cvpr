"""Early, exploratory paired full-val audit for one exact512 n=4 seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_direction_eval import summarize
from analyze_matched_pair import compare, load_label


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full_path = args.root / "manifest.json"
    if digest(full_path) != FULL_SHA or \
            digest(args.screen_manifest) != SCREEN_SHA:
        raise ValueError("frozen val-unseen manifest mismatch")
    full = json.loads(full_path.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = [str(eid) for eid in full["episode_ids"]]
    scenes = [str(scene) for scene in full["scene_ids"]]
    screen_ids = {str(eid) for eid in screen["episode_ids"]}
    if len(ids) != 1839 or len(set(ids)) != 1839 or \
            len(scenes) != 1839 or len(set(scenes)) != 11 or \
            len(screen_ids) != 256 or not screen_ids.issubset(ids):
        raise ValueError("full or screen episode coverage mismatch")
    candidate_name = f"oracle_turnwise_exact512_128_seed{args.seed}"
    control_name = f"oracle_exact512_control_128_seed{args.seed}"
    candidate = load_label(args.root, candidate_name, ids, 4)
    control = load_label(args.root, control_name, ids, 4)
    scopes = {}
    for name, members in (
            ("full", set(ids)),
            ("reused_screen", screen_ids),
            ("outside_reused_screen", set(ids) - screen_ids)):
        scope_ids = [eid for eid in ids if eid in members]
        scope_scenes = [scene for eid, scene in zip(ids, scenes)
                        if eid in members]
        expected = {"full": 1839, "reused_screen": 256,
                    "outside_reused_screen": 1583}[name]
        if len(scope_ids) != expected:
            raise ValueError(f"wrong {name} size")
        cm = summarize(candidate, scope_ids)
        bm = summarize(control, scope_ids)
        if cm["count"] != expected or bm["count"] != expected or \
                cm["inference_errors"] or bm["inference_errors"]:
            raise ValueError(f"coverage or inference error in {name}")
        scopes[name] = {
            "episodes": expected, "scenes": len(set(scope_scenes)),
            "candidate_metrics": cm, "control_metrics": bm,
            "paired": compare(candidate, control, scope_ids, scope_scenes),
        }
    result = {
        "schema": "oracle_exact512_one_seed_full_val_unseen_v1",
        "seed": args.seed,
        "candidate": candidate_name, "control": control_name,
        "full_manifest_sha256": FULL_SHA,
        "screen_manifest_sha256": SCREEN_SHA,
        "candidate_validator_sha256": digest(
            args.root / f"{candidate_name}.validated.json"),
        "control_validator_sha256": digest(
            args.root / f"{control_name}.validated.json"),
        **scopes,
        "interpretation": (
            "One paired seed and one decode per episode; not the three-seed "
            "result. The reused 256-item screen and its complement are "
            "development data. Candidate training used privileged simulator "
            "distance, not an observation-only learned reward."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result[name]["paired"] for name in scopes}, indent=2))


if __name__ == "__main__":
    main()
