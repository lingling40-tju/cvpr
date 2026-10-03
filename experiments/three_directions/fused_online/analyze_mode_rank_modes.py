"""Audit episode coverage and termination behavior in the mode-rank pilot.

The termination comparison is descriptive; it cannot establish why the
candidate's navigation success changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


MANIFEST_SHA256 = "1d81cdd30676cadeaa7cd5a59ae6fc5905af99d62dec8ad0ab62afa93f6f19fc"


def load_arm(root: Path, label: str, ids: list[str]) -> dict[str, dict]:
    if not (root / f"{label}.completed").is_file():
        raise ValueError(f"incomplete arm: {label}")
    rows = {}
    for shard in range(4):
        folder = root / label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        expected = ids[shard::4]
        observed = list(map(str, summary["episode_ids"]))
        if (summary["count"] != len(expected) or set(observed) != set(expected)
                or len(observed) != len(set(observed)) or summary["inference_errors"]):
            raise ValueError(f"invalid shard: {label}/{shard}")
        for eid in expected:
            row = json.loads((folder / "log" / f"stats_{eid}_0.json").read_text())
            if str(row["id"]) != eid or row.get("early_stop_reason") == "inference_error":
                raise ValueError(f"invalid episode: {label}/{eid}")
            rows[eid] = row
    if set(rows) != set(ids):
        raise ValueError(f"coverage mismatch: {label}")
    return rows


def arm_summary(rows: dict[str, dict], ids: list[str]) -> dict:
    failures = [eid for eid in ids if not rows[eid]["success"]]
    return {
        "count": len(ids),
        "successes": len(ids) - len(failures),
        "oracle_successes": sum(bool(rows[eid]["oracle_success"]) for eid in ids),
        "termination_reasons_all": dict(Counter(str(rows[eid].get("early_stop_reason")) for eid in ids)),
        "termination_reasons_failures": dict(Counter(str(rows[eid].get("early_stop_reason")) for eid in failures)),
        "mean_path_length_m": statistics.mean(float(rows[eid]["path_length"]) for eid in ids),
        "mean_distance_to_goal_m": statistics.mean(float(rows[eid]["distance_to_goal"]) for eid in ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual_hash = hashlib.sha256((args.eval_root / "manifest.json").read_bytes()).hexdigest()
    if actual_hash != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads((args.eval_root / "manifest.json").read_text())
    ids = list(map(str, manifest["episode_ids"]))
    if len(ids) != 256 or len(set(ids)) != 256 or len(set(manifest["scene_ids"])) != 9:
        raise ValueError("unexpected fixed evaluation set")
    candidate_label = "mode_rank_group4_64_seed11"
    control_label = "group4_64_seed11_third256"
    candidate = load_arm(args.eval_root, candidate_label, ids)
    control = load_arm(args.eval_root, control_label, ids)
    groups = {
        "candidate_only_successes": [eid for eid in ids if candidate[eid]["success"] and not control[eid]["success"]],
        "control_only_successes": [eid for eid in ids if control[eid]["success"] and not candidate[eid]["success"]],
        "both_failures": [eid for eid in ids if not candidate[eid]["success"] and not control[eid]["success"]],
    }
    paired = {}
    for name, members in groups.items():
        paired[name] = {
            "count": len(members),
            "candidate_failure_reasons": dict(Counter(
                str(candidate[eid].get("early_stop_reason")) for eid in members
                if not candidate[eid]["success"])),
            "control_failure_reasons": dict(Counter(
                str(control[eid].get("early_stop_reason")) for eid in members
                if not control[eid]["success"])),
        }
    result = {
        "schema": "mode_rank_val256_termination_audit_v1",
        "manifest_sha256": actual_hash,
        "interpretation": "Null reason on a final STOP action means neither the turn/step limits nor the evaluator's rotation guard forced it. This is a post-hoc description, not a causal explanation.",
        "candidate": candidate_label,
        "control": control_label,
        "candidate_metrics": arm_summary(candidate, ids),
        "control_metrics": arm_summary(control, ids),
        "paired_outcome_groups": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
