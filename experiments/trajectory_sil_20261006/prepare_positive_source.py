"""Install the frozen positive-trajectory estimator into an isolated checkout.

Run only on a new copy of the ActiveVLN source tree. The script verifies the
exact trainer source before editing and refuses to touch the live n=4 tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


TRAINER_SHA256 = "ca3e7ec596f4c5cc13b6b574a3f71cd9040db6a34090e8776f2ce44a8288354b"
LIVE_SOURCE_NAME = "ActiveVLN_norm_terminal_rloo_20261006"
ANCHOR = '''        if oracle_turnwise:
            if os.environ.get("VLN_TURN_RLOO") == "1":'''
REPLACEMENT = '''        if os.environ.get("VLN_POSITIVE_TRAJECTORY") == "1":
            if oracle_turnwise:
                raise ValueError("positive trajectory update cannot use oracle turnwise rewards")
            from verl.trainer.ppo.positive_trajectory_advantage import (
                positive_trajectory_advantage,
            )
            advantages, returns = positive_trajectory_advantage(
                terminal_score=data.batch["token_level_rewards"].sum(dim=-1),
                task_success=data.batch["task_success"].reshape(-1),
                action_mask=grpo_calculation_mask,
                group_ids=[str(value) for value in data.non_tensor_batch["uid"]],
            )
        elif oracle_turnwise:
            if os.environ.get("VLN_TURN_RLOO") == "1":'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.target_root.resolve()
    if root.name == LIVE_SOURCE_NAME:
        raise ValueError("refusing to edit the live n=4 source tree")
    trainer = root / "verl/trainer/ppo/ray_trainer.py"
    destination = root / "verl/trainer/ppo/positive_trajectory_advantage.py"
    source = Path(__file__).with_name("positive_trajectory_advantage.py")
    if trainer.is_symlink() or destination.is_symlink():
        raise ValueError("trainer and new estimator must be real isolated files")
    if digest(trainer) != TRAINER_SHA256 or destination.exists():
        raise ValueError("trainer source differs or estimator already installed")
    text = trainer.read_text()
    if text.count(ANCHOR) != 1:
        raise ValueError("trainer insertion anchor is absent or ambiguous")
    updated = text.replace(ANCHOR, REPLACEMENT)
    compile(updated, str(trainer), "exec")
    compile(source.read_text(), str(destination), "exec")
    trainer.write_text(updated)
    shutil.copy2(source, destination)
    print(json.dumps({"source_trainer_sha256": TRAINER_SHA256,
                      "patched_trainer_sha256": digest(trainer),
                      "estimator_sha256": digest(destination)}))


if __name__ == "__main__":
    main()
