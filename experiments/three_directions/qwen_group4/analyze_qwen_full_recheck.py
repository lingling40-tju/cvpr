"""Audit a post hoc one-seed Qwen full-val sensitivity check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_direction_eval import summarize
from analyze_matched_pair import compare, load_label


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--screen-manifest", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--full-pair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    full_manifest = args.root / "manifest.json"
    if digest(full_manifest) != FULL_SHA or digest(args.screen_manifest) != SCREEN_SHA:
        raise ValueError("frozen manifest hash mismatch")
    full = json.loads(full_manifest.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = [str(x) for x in full["episode_ids"]]
    scenes = [str(x) for x in full["scene_ids"]]
    screen_ids = {str(x) for x in screen["episode_ids"]}
    if len(ids) != 1839 or len(set(ids)) != 1839 or len(scenes) != 1839:
        raise ValueError("full manifest coverage mismatch")
    if len(screen_ids) != 256 or not screen_ids.issubset(ids):
        raise ValueError("screen manifest coverage mismatch")
    candidate = load_label(args.root, args.candidate, ids, 4)
    control = load_label(args.root, args.control, ids, 4)
    full_pair = json.loads(args.full_pair.read_text())
    if full_pair["candidate"] != args.candidate or full_pair["control"] != args.control:
        raise ValueError("full pair labels mismatch")
    if full_pair["manifest_sha256"] != FULL_SHA or full_pair["episodes"] != 1839:
        raise ValueError("full pair manifest mismatch")
    outside_ids = [eid for eid in ids if eid not in screen_ids]
    outside_scenes = [scene for eid, scene in zip(ids, scenes) if eid not in screen_ids]
    if len(outside_ids) != 1583:
        raise ValueError("screen complement mismatch")

    scopes = {}
    for name, scope_ids, scope_scenes in (
        ("full", ids, scenes),
        ("outside_reused_screen", outside_ids, outside_scenes),
    ):
        candidate_metrics = summarize(candidate, scope_ids)
        control_metrics = summarize(control, scope_ids)
        if candidate_metrics["inference_errors"] or control_metrics["inference_errors"]:
            raise ValueError("inference errors")
        scopes[name] = {
            "episodes": len(scope_ids),
            "scenes": len(set(scope_scenes)),
            "candidate_metrics": candidate_metrics,
            "control_metrics": control_metrics,
            "paired": compare(candidate, control, scope_ids, scope_scenes),
        }
    for key in ("sr_pp", "spl_pp"):
        if not math.isclose(scopes["full"]["paired"][key], full_pair["paired"][key], abs_tol=1e-10):
            raise ValueError(f"full pair {key} recomputation mismatch")
    for arm in ("candidate_metrics", "control_metrics"):
        for key in ("count", "successes", "sr", "spl"):
            if not math.isclose(scopes["full"][arm][key], full_pair[arm][key], abs_tol=1e-10):
                raise ValueError(f"full pair {arm}.{key} recomputation mismatch")
    report = {
        "schema": "qwen_group_rank_one_seed_posthoc_full_recheck_v1",
        "candidate": args.candidate,
        "control": args.control,
        "full_manifest_sha256": FULL_SHA,
        "screen_manifest_sha256": SCREEN_SHA,
        "full_pair_sha256": digest(args.full_pair),
        "full": scopes["full"],
        "outside_reused_screen": scopes["outside_reused_screen"],
        "interpretation": (
            "One seed, one decode per checkpoint and episode. The full recheck was chosen "
            "after inspecting a negative 256-episode screen; its complement excludes "
            "that screen but is still development data. Scene bootstrap is exploratory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k]["paired"] for k in ("full", "outside_reused_screen")}, indent=2))


if __name__ == "__main__":
    main()
