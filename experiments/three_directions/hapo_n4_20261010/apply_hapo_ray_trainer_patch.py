#!/usr/bin/env python3
"""Apply one exact HAPO dispatch patch to the verified ReMax trainer copy."""

import hashlib
import os
import sys
from pathlib import Path

EXPECTED_SHA256 = "43a26733db744448ebc1365ffca2ae6eb495a0a94c5f9923da4425c6f8a550ec"
NEEDLE = '''        elif oracle_turnwise:
            if os.environ.get("VLN_TURN_RLOO") == "1":
'''
REPLACEMENT = '''        elif oracle_turnwise:
            if os.environ.get("VLN_HAPO") == "1":
                from verl.trainer.ppo.hapo_group4_advantage import (
                    from_activevln_batch as hapo_from_batch,
                )
                advantages, returns = hapo_from_batch(data, grpo_calculation_mask)
            elif os.environ.get("VLN_TURN_RLOO") == "1":
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_hapo_ray_trainer_patch.py /isolated/verl/trainer/ppo/ray_trainer.py")
    path = Path(sys.argv[1])
    if not path.is_file() or sha(path) != EXPECTED_SHA256:
        raise SystemExit("refusing to patch: trainer source is absent or differs from frozen baseline")
    source = path.read_text()
    if source.count(NEEDLE) != 1:
        raise SystemExit("refusing to patch: expected one exact HAPO insertion point")
    output = source.replace(NEEDLE, REPLACEMENT, 1)
    tmp = path.with_name(path.name + f".hapo-tmp-{os.getpid()}")
    tmp.write_text(output)
    os.replace(tmp, path)
    print(sha(path))


if __name__ == "__main__":
    main()
