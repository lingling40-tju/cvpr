"""Integrate the preregistered normalized RLOO into a separate source tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

TREE_NAME = "ActiveVLN_norm_terminal_rloo_20261006"
EXPECTED_ADAPTER_SHA256 = "08506defe64cc58aac8f36389dcf3956372fea2361b964a14f89155c2afd9fcd"
EXPECTED_MODULE_SHA256 = "772d00a438d32a716b2c666d681e40af8794375d5e075b381194c2006cf80fa1"
OLD = '''    outcome_reward = data.batch["token_level_rewards"].sum(dim=-1)
    return turn_rloo_advantage(
'''
NEW = '''    outcome_reward = data.batch["token_level_rewards"].sum(dim=-1)
    if os.environ.get("VLN_NORMALIZED_TERMINAL_RLOO") == "1":
        from verl.trainer.ppo.normalized_terminal_rloo import (
            normalized_terminal_turn_rloo,
        )
        return normalized_terminal_turn_rloo(
            outcome_reward, turn_valid, action_turn_index,
            [str(x) for x in group_ids],
        )
    return turn_rloo_advantage(
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    tree, source = map(Path, sys.argv[1:3])
    if tree.name != TREE_NAME:
        raise ValueError("refusing to modify an active or unexpected source tree")
    adapter = tree / "verl/trainer/ppo/turn_rloo_advantage.py"
    module = tree / "verl/trainer/ppo/normalized_terminal_rloo.py"
    before = adapter.read_bytes()
    module_source = source.read_bytes()
    if sha256(before) != EXPECTED_ADAPTER_SHA256 or sha256(module_source) != EXPECTED_MODULE_SHA256:
        raise ValueError("source hashes differ from the pre-result specification")
    text = before.decode()
    if text.count(OLD) != 1 or text.count("from collections import defaultdict\n") != 1:
        raise ValueError("expected adapter insertion points changed")
    if module.exists():
        raise ValueError("normalized module already exists")
    text = text.replace("from collections import defaultdict\n",
                        "from collections import defaultdict\nimport os\n")
    text = text.replace(OLD, NEW)
    adapter.write_text(text)
    module.write_bytes(module_source)
    audit = {
        "schema": "normalized_terminal_source_patch_v1",
        "adapter_before_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_after_sha256": sha256(adapter.read_bytes()),
        "normalized_module_sha256": sha256(module.read_bytes()),
        "tree": str(tree),
        "active_factorial_source_untouched": True,
    }
    (tree / "runlogs/normalized_source_patch.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
