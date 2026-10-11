"""Check one completed inference sweep before declaring its label complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_train_scene_pair import SCREEN_HASHES, digest, load_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(SCREEN_HASHES), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.manifest) != SCREEN_HASHES[args.role]:
        raise ValueError("frozen train-scene manifest SHA-256 mismatch")
    manifest = json.loads(args.manifest.read_text())
    ids = [str(item) for item in manifest["episode_ids"]]
    if manifest["role"] != args.role or manifest["split"] != "train" or \
            len(ids) != 256 or len(set(ids)) != 256:
        raise ValueError("train-scene manifest differs")
    rows = load_arm(args.root, args.label, ids, require_completed=False)
    result = {"role": args.role, "label": args.label, "episodes": len(rows),
              "successes": sum(row[0] for row in rows.values()),
              "spl": sum(row[1] for row in rows.values()) / len(rows),
              "inference_errors": 0, "manifest_sha256": SCREEN_HASHES[args.role]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
