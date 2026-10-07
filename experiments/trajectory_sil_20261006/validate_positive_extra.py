"""Validate frozen episode identity and raw metrics before a completion marker."""

import argparse
import hashlib
import json
import math
from pathlib import Path


SCREENS = {
    "development": ("8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3", 256, 8, "train"),
    "reserved": ("412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241", 256, 8, "train"),
    "val_unseen": ("262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e", 1839, 11, "val_unseen"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen(path, role):
    digest, n, scenes, split = SCREENS[role]
    if sha(path) != digest:
        raise ValueError("frozen manifest hash differs")
    m = json.loads(path.read_text())
    ids = [str(i) for i in m["episode_ids"]]
    if m["split"] != split or len(ids) != len(set(ids)) or len(ids) != n or \
            len(m["scene_ids"]) != n or len(set(m["scene_ids"])) != scenes:
        raise ValueError("frozen episode/scene coverage differs")
    if role != "val_unseen" and m["role"] != role:
        raise ValueError("screen role differs")
    return m, ids


def load_metrics(root, label, ids, split, require_completed=True):
    if require_completed and ((root / f"{label}.failed").exists() or
                              not (root / f"{label}.completed").exists()):
        raise ValueError(f"label not complete: {label}")
    rows = {}
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}"
        subset = ids[shard::4]
        expected = {f"stats_{i}_0.json" for i in subset}
        if {p.name for p in (folder / "log").glob("stats_*_0.json")} != expected:
            raise ValueError(f"raw shard coverage differs: {label} {shard}")
        summary = json.loads((folder / "summary.json").read_text())
        # Habitat may reorder its episode list during iteration. Shard
        # membership and uniqueness are fixed; traversal order is not.
        summary_ids = [str(i) for i in summary["episode_ids"]]
        if summary["label"] != label or summary["split"] != split or \
                len(summary_ids) != len(set(summary_ids)) or set(summary_ids) != set(subset) or \
                summary["count"] != len(subset) or \
                summary["inference_errors"] != 0 or summary["max_turns"] != 12:
            raise ValueError(f"shard summary identity differs: {label} {shard}")
        successes, spl_sum = 0, 0.0
        for i in subset:
            raw = json.loads((folder / "log" / f"stats_{i}_0.json").read_text())
            if str(raw.get("id")) != i or raw.get("early_stop_reason") == "inference_error":
                raise ValueError(f"raw episode identity or inference error: {label} {i}")
            success, spl = raw.get("success"), raw.get("spl")
            if success not in (0, 0.0, 1, 1.0) or isinstance(spl, bool) or \
                    not isinstance(spl, (int, float)) or not math.isfinite(spl) or not 0 <= spl <= 1:
                raise ValueError(f"invalid raw metric: {label} {i}")
            for key in ("distance_to_goal", "path_length"):
                x = raw.get(key)
                if isinstance(x, bool) or not isinstance(x, (int, float)) or \
                        not math.isfinite(x) or x < 0:
                    raise ValueError(f"invalid {key}: {label} {i}")
            rows[i] = {"success": int(success), "spl": float(spl),
                       "distance_to_goal": raw["distance_to_goal"],
                       "path_length": raw["path_length"],
                       "early_stop_reason": raw.get("early_stop_reason")}
            successes += int(success)
            spl_sum += spl
        if summary["successes"] != successes or not math.isclose(
                summary["spl"], spl_sum / len(subset), rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"raw and shard summary disagree: {label} {shard}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--role", choices=SCREENS, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    m, ids = screen(args.manifest, args.role)
    rows = load_metrics(args.root, args.label, ids, m["split"], require_completed=False)
    out = {"schema": "positive_extra_raw_coverage_v1", "role": args.role,
           "label": args.label, "episodes": len(rows), "scenes": len(set(m["scene_ids"])),
           "successes": sum(r["success"] for r in rows.values()),
           "spl": sum(r["spl"] for r in rows.values()) / len(rows),
           "inference_errors": 0, "raw_episode_id_checked": True,
           "manifest_sha256": sha(args.manifest)}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
