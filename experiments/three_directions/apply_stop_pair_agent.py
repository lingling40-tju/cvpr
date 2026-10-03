"""Checksum-guarded STOP-pair hook for an isolated ActiveVLN copy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA = "c5095a3f3e73a27357b8597bac72f7256434669e7c0f38b7f4a262852b374256"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"agent hook anchor count {source.count(old)}: {old[:65]}")
    return source.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    agent = args.root / "verl/workers/agent/parallel_env_vlnce.py"
    if digest(agent) != SOURCE_SHA:
        raise ValueError("unexpected or already patched source agent")
    source = agent.read_text()
    changes = (
        ("from .qwen_group_reward import group_relative_adjustments",
         "from .stop_pair_group4_reward import group_relative_adjustments"),
        ('config.agent.reward.get("qwen_group_relative", False)',
         'config.agent.reward.get("stop_pair_group4", False)'),
        ('info_list[index]["reward_components"]["qwen_group_ordinal"]',
         'info_list[index]["reward_components"]["stop_pair_ordinal"]'),
        ('info_list[index]["fused_reward"]["removed_bonus"] = old_bonus\n'
         '            info_list[index]["fused_reward"]["applied_ordinal"] = new_ordinal',
         'info_list[index]["stop_pair_diagnostic"] = {\n'
         '                "removed_bonus": old_bonus, "applied_ordinal": new_ordinal}'),
        ('[qwen group ordinal]', '[stop pair ordinal]'),
    )
    for old, new in changes:
        source = replace_once(source, old, new)
    agent.write_text(source)
    print("agent_sha256", digest(agent))


if __name__ == "__main__":
    main()
