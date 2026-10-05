"""Fail-closed patch of the isolated ActiveVLN oracle trainer for turn RLOO."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


EXPECTED = "2a135d65f35d0a5d9108404746ff102e69afac171b9ce6dc6764fd9c3bf10393"
OLD = """        if oracle_turnwise:
            advantages, returns = from_activevln_batch(data, grpo_calculation_mask)
"""
NEW = """        if oracle_turnwise:
            if os.environ.get("VLN_TURN_RLOO") == "1":
                from verl.trainer.ppo.turn_rloo_advantage import (
                    from_activevln_batch as turn_rloo_from_batch,
                )
                advantages, returns = turn_rloo_from_batch(data, grpo_calculation_mask)
            else:
                advantages, returns = from_activevln_batch(data, grpo_calculation_mask)
"""


def main() -> None:
    target = Path(sys.argv[1])
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError("trainer source hash changed; refusing patch")
    source = data.decode()
    if source.count(OLD) != 1:
        raise ValueError("expected oracle branch not found exactly once")
    target.write_text(source.replace(OLD, NEW))
    print(hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
