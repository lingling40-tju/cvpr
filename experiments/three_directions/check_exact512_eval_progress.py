"""Count exact val-unseen episode stats, excluding extra-info JSON files.

Partial counts are progress only. A completed label is accepted only with
the frozen 1,839-ID manifest, exact shard coverage, and zero-error validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


MANIFEST_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
STATS = re.compile(r"stats_(.+)_0\.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root: Path, label: str, ids: list[str]) -> dict:
    expected = set(ids)
    directory = root / label
    found_all: set[str] = set()
    per_shard = []
    for shard in range(4):
        folder = directory / f"shard_{shard:02d}" / "log"
        paths = list(folder.glob("stats_*_0.json"))
        found = []
        for path in paths:
            match = STATS.fullmatch(path.name)
            if not match:
                raise ValueError(f"unexpected stats name: {path}")
            found.append(match.group(1))
        shard_expected = set(ids[shard::4])
        if len(found) != len(set(found)) or \
                not set(found).issubset(shard_expected) or \
                found_all.intersection(found):
            raise ValueError(f"duplicate or off-shard stats: {label}/{shard}")
        found_all.update(found)
        per_shard.append(len(found))
    if not found_all.issubset(expected):
        raise ValueError("unknown val-unseen episode")
    completed = (root / f"{label}.completed").is_file()
    failed = (root / f"{label}.failed").is_file()
    validated_path = root / f"{label}.validated.json"
    report = {"label": label, "completed": completed, "failed": failed,
              "unique_stats": len(found_all), "expected": len(ids),
              "missing": len(expected - found_all), "per_shard": per_shard}
    if completed:
        if failed or len(found_all) != len(ids) or not validated_path.is_file():
            raise ValueError(f"completed label lacks exact validated coverage: {label}")
        validated = json.loads(validated_path.read_text())
        if validated.get("label") != label or \
                validated.get("episodes") != len(ids) or \
                validated.get("inference_errors") != 0:
            raise ValueError(f"invalid final validator: {label}")
        report["validated_successes"] = int(validated["successes"])
        report["validator_sha256"] = digest(validated_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()
    manifest_path = args.root / "manifest.json"
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError("frozen complete val-unseen manifest changed")
    manifest = json.loads(manifest_path.read_text())
    ids = [str(value) for value in manifest["episode_ids"]]
    if len(ids) != 1839 or len(set(ids)) != 1839:
        raise ValueError("incomplete or duplicate val-unseen manifest")
    print(json.dumps({"schema": "exact512_eval_progress_v1",
                      "manifest_sha256": MANIFEST_SHA,
                      "labels": [inspect(args.root, label, ids)
                                 for label in args.labels]}, indent=2))


if __name__ == "__main__":
    main()
