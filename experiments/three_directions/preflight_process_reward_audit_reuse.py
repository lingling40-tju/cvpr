"""Count prospective audit episode IDs already present in old rollouts.

Reads only episode IDs and source checksums. It does not inspect model
scores, simulator distances, success labels, images, or reward targets.
The frozen 123-episode audit selection is never altered by this count.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


EXPECTED_AUDIT_SHA = "be0d2f8df0138a9ccd13109f7a21e187dbbf2aab44817250a81aa961da2dccc5"
EXPECTED_SOURCE_SHA = "aa32f68a906f952b63bc57bffdd0aa0bd0e3b9108d932266533bd597c58c9681"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.audit_manifest) != EXPECTED_AUDIT_SHA or \
            digest(args.source_manifest) != EXPECTED_SOURCE_SHA:
        raise ValueError("frozen manifest checksum mismatch")
    audit = json.loads(args.audit_manifest.read_text())
    source = json.loads(args.source_manifest.read_text())
    if audit["schema"] != "process_reward_prospective_train_scene_audit_v1" or \
            audit["episode_count"] != 123 or audit["scene_count"] != 7 or \
            source["schema"] != "policy_process_train_manifest_v1":
        raise ValueError("manifest schema or coverage changed")
    audit_scene = {row["episode_id"]: row["scene_id"]
                   for row in audit["episodes"]}
    if len(audit_scene) != 123:
        raise ValueError("duplicate prospective audit episode")
    seen = Counter()
    for seed, plan in sorted(source["sources"].items()):
        path = args.source_root / plan["path"]
        if digest(path) != plan["sha256"]:
            raise ValueError(f"source rollout checksum mismatch seed {seed}")
        steps, total = [], 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                steps.append(int(row["step"]))
                for info in row["info"]:
                    total += 1
                    eid = str(info["episode_id"])
                    if eid in audit_scene:
                        seen[eid] += 1
        if steps != list(range(1, 129)) or total != plan["rollouts"]:
            raise ValueError(f"incomplete source rollout seed {seed}")
    reused = sorted((eid for eid in audit_scene if seen[eid]), key=int)
    missing = sorted((eid for eid in audit_scene if not seen[eid]), key=int)
    if len(reused) != 8 or len(missing) != 115:
        raise ValueError("unexpected prospective audit overlap")
    report = {
        "schema": "process_reward_audit_source_reuse_preflight_v1",
        "source_sha256": {"prospective_audit": EXPECTED_AUDIT_SHA,
                          "prior_rollout_manifest": EXPECTED_SOURCE_SHA},
        "total_audit_episodes": len(audit_scene),
        "existing_rollout_episode_ids": reused,
        "existing_rollout_trajectories": sum(seen[eid] for eid in reused),
        "episodes_requiring_new_rollout": len(missing),
        "missing_episode_ids": missing,
        "existing_by_scene": dict(sorted(Counter(audit_scene[eid]
                                                  for eid in reused).items())),
        "missing_by_scene": dict(sorted(Counter(audit_scene[eid]
                                                 for eid in missing).items())),
        "interpretation": (
            "ID-only coverage preflight. The audit selection remains all "
            "123 frozen train episodes. Repeated trajectories do not increase "
            "the unique-episode count. No labels or model predictions were read."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reused_unique_episodes": len(reused),
                      "reused_trajectories": report["existing_rollout_trajectories"],
                      "new_unique_episodes_needed": len(missing),
                      "output_sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
