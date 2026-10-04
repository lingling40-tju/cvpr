"""Independently audit a post-screen one-seed oracle full-val recheck."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--full-pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full_manifest = args.root / "manifest.json"
    if digest(full_manifest) != FULL_SHA or \
            digest(args.screen_manifest) != SCREEN_SHA:
        raise ValueError("frozen full/screen manifest hash mismatch")
    full = json.loads(full_manifest.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = list(map(str, full["episode_ids"]))
    scenes = list(map(str, full["scene_ids"]))
    screen_ids = set(map(str, screen["episode_ids"]))
    if len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != 1839 or \
            len(set(scenes)) != 11 or len(screen_ids) != 256 or \
            not screen_ids.issubset(ids):
        raise ValueError("full or screen episode coverage invalid")
    candidate = load_label(args.root, args.candidate, ids, 4)
    control = load_label(args.root, args.control, ids, 4)
    pair = json.loads(args.full_pair.read_text())
    if pair["candidate"] != args.candidate or pair["control"] != args.control or \
            pair["manifest_sha256"] != FULL_SHA or pair["episodes"] != 1839:
        raise ValueError("paired full report identity mismatch")
    outside = [(eid, scene) for eid, scene in zip(ids, scenes)
               if eid not in screen_ids]
    if len(outside) != 1583:
        raise ValueError("reused screen complement mismatch")
    scopes = {}
    for name, scope_ids, scope_scenes in (
            ("full", ids, scenes),
            ("outside_reused_screen", [x[0] for x in outside],
             [x[1] for x in outside])):
        cand = summarize(candidate, scope_ids)
        ctrl = summarize(control, scope_ids)
        if cand["inference_errors"] or ctrl["inference_errors"]:
            raise ValueError("inference errors in paired full recheck")
        scopes[name] = {"episodes": len(scope_ids),
                        "scenes": len(set(scope_scenes)),
                        "candidate_metrics": cand,
                        "control_metrics": ctrl,
                        "paired": compare(candidate, control,
                                          scope_ids, scope_scenes)}
    for key in ("sr_pp", "spl_pp"):
        if not math.isclose(scopes["full"]["paired"][key],
                            pair["paired"][key], abs_tol=1e-10):
            raise ValueError(f"full paired {key} recomputation mismatch")
    for arm in ("candidate_metrics", "control_metrics"):
        for key in ("count", "successes", "sr", "spl"):
            if not math.isclose(scopes["full"][arm][key], pair[arm][key],
                                abs_tol=1e-10):
                raise ValueError(f"full {arm}.{key} recomputation mismatch")
    result = {
        "schema": "oracle_turnwise_one_seed_postscreen_full_recheck_v1",
        "candidate": args.candidate, "control": args.control,
        "full_manifest_sha256": FULL_SHA,
        "screen_manifest_sha256": SCREEN_SHA,
        "full_pair_sha256": digest(args.full_pair),
        "full": scopes["full"],
        "outside_reused_screen": scopes["outside_reused_screen"],
        "interpretation": (
            "One seed, one decode per checkpoint and episode. Full evaluation was "
            "chosen after a positive 256-episode screen; the 1,583-episode complement "
            "is still development data. This reward uses privileged simulator distance "
            "during training, not a learned semantic verifier. Scene bootstrap is exploratory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key]["paired"]
                      for key in ("full", "outside_reused_screen")}, indent=2))


if __name__ == "__main__":
    main()
