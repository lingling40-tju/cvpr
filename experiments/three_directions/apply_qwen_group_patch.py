"""Checksum-guarded six-view Qwen group reward hook for an isolated copy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
    "verl/workers/agent/parallel_env_vlnce.py":
        "5a35ed8f48337b8362a44e6ef31e5e4ed546c481372aba8c6ab6d72433c094fa",
    "vlnce_server/semantic_reward/env.py":
        "04ab342e477c1353d46446a0e64f51894ce7ac263607f01967719707defd27f9",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"patch anchor mismatch ({text.count(old)}): {old[:70]}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    source = {}
    for relative, expected in EXPECTED.items():
        path = args.root / relative
        if digest(path) != expected:
            raise ValueError(f"source changed or already patched: {relative}")
        source[relative] = path.read_text()

    path = "vlnce_server/semantic_reward/env.py"
    text = source[path]
    text = replace_once(
        text,
        'indices = [round((len(self._fused_views) - 1) * i / 3) for i in range(4)]',
        'indices = [math.floor((len(self._fused_views) - 1) * i / 5 + 0.5) '
        'for i in range(6)]')
    text = replace_once(text, 'view.thumbnail((336, 336))',
                        'view.thumbnail((448, 448))')
    text = replace_once(
        text,
        'json={"instruction": self.instruction, "images": encoded,\n'
        '                      "initial": [index == 0 for index in indices]},',
        'json={"episode_id": self.config.episode_id,\n'
        '                      "instruction": self.instruction, "images": encoded,\n'
        '                      "initial": [index == 0 for index in indices]},')
    source[path] = text

    path = "verl/workers/agent/parallel_env_vlnce.py"
    text = source[path]
    text = replace_once(text,
        'from .mode_stratified_reward import mode_stratified_adjustments',
        'from .qwen_group_reward import group_relative_adjustments')
    text = replace_once(text,
        'config.agent.reward.get("mode_stratified_ordinal", False)',
        'config.agent.reward.get("qwen_group_relative", False)')
    text = replace_once(text, 'mode_stratified_adjustments(\n',
                        'group_relative_adjustments(\n')
    text = replace_once(text,
                        'info_list[index]["reward_components"]["mode_ordinal"]',
                        'info_list[index]["reward_components"]["qwen_group_ordinal"]')
    text = replace_once(text, '[mode stratified ordinal]',
                        '[qwen group ordinal]')
    source[path] = text

    for relative, patched in source.items():
        target = args.root / relative
        backup = target.with_name(target.name + ".qwen_group_source")
        backup.write_bytes(target.read_bytes())
        target.write_text(patched)
        print(relative, "patched_sha256", digest(target))
    helper = args.root / "verl/workers/agent/qwen_group_reward.py"
    helper.write_bytes(args.helper.read_bytes())
    print("helper_sha256", digest(helper))


if __name__ == "__main__":
    main()
