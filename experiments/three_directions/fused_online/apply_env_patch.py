"""Install the terminal fused-reward hook in an isolated ActiveVLN copy.

This patch is intentionally checksum-guarded. The original 2026-10-02
control environment and its completed checkpoints are never modified.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


EXPECTED = {
    "vlnce_server/env_config.py": "d3f8dd8553d71bf61a830b19898df7a8c2d207ed3b614fbe11066e2c6c6bff47",
    "verl/workers/agent/parallel_env_vlnce.py": "90d66792129a4e9fab29aa3ee247ceb86a6216197a668bbf03640c453107aa1e",
    "vlnce_server/semantic_reward/env.py": "d9cf6db6d0e098c707f2df261d3f6e39d36bf7a1f8d617934357e10814a6d057",
}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"patch anchor expected once, found {text.count(old)}: {old[:80]}")
    return text.replace(old, new, 1)


def patch(root: Path) -> None:
    source = {}
    for relative, checksum in EXPECTED.items():
        contents = (root / relative).read_bytes()
        if hashlib.sha256(contents).hexdigest() != checksum:
            raise ValueError(f"upstream source changed: {relative}")
        source[relative] = contents.decode()

    path = "vlnce_server/env_config.py"
    text = source[path]
    text = replace_once(text,
        '    semantic_event_manifest: str = ""\n',
        '    semantic_event_manifest: str = ""\n'
        '    fused_reward_weight: float = 0.0\n'
        '    fused_reward_url: str = "http://127.0.0.1:8021"\n')
    source[path] = text

    path = "verl/workers/agent/parallel_env_vlnce.py"
    text = source[path]
    text = replace_once(text,
        '                    semantic_event_manifest=self.config.reward.get("semantic_event_manifest", ""),\n',
        '                    semantic_event_manifest=self.config.reward.get("semantic_event_manifest", ""),\n'
        '                    fused_reward_weight=self.config.reward.get("fused_reward_weight", 0.0),\n'
        '                    fused_reward_url=self.config.reward.get("fused_reward_url", "http://127.0.0.1:8021"),\n')
    source[path] = text

    path = "vlnce_server/semantic_reward/env.py"
    text = source[path]
    text = replace_once(text, 'import math\nimport json\n',
                        'import math\nimport json\nimport base64\nimport io\nimport requests\nfrom PIL import Image\n')
    text = replace_once(text,
        '        self.success_floor = float(config.get("semantic_success_floor", 0.0))\n',
        '        self.success_floor = float(config.get("semantic_success_floor", 0.0))\n'
        '        self.fused_weight = float(config.get("fused_reward_weight", 0.0))\n'
        '        self.fused_url = str(config.get("fused_reward_url", "http://127.0.0.1:8021"))\n'
        '        if not 0 <= self.fused_weight <= 1:\n'
        '            raise ValueError("fused reward weight must be in [0,1]")\n'
        '        self._fused_views = []\n')
    text = replace_once(text,
        '        self._last_image = observations[-1]["multi_modal_data"]["<image>"][0]\n',
        '        self._last_image = observations[-1]["multi_modal_data"]["<image>"][0]\n'
        '        self._fused_views = [self._last_image.copy()] if self.fused_weight > 0 else []\n')
    text = replace_once(text,
        '        after_image = observation["multi_modal_data"]["<image>"][0]\n',
        '        after_image = observation["multi_modal_data"]["<image>"][0]\n'
        '        if self.fused_weight > 0:\n'
        '            self._fused_views.append(after_image.copy())\n')
    text = replace_once(text,
        '        floor_reward = self.success_floor if info.get("task_success") else 0.0\n'
        '        added_reward = semantic_reward + floor_reward\n',
        '        floor_reward = self.success_floor if info.get("task_success") else 0.0\n'
        '        fused_bonus = 0.0\n'
        '        fused_result = {"status": "disabled"}\n'
        '        if done and self.fused_weight > 0:\n'
        '            indices = [round((len(self._fused_views) - 1) * i / 3) for i in range(4)]\n'
        '            encoded = []\n'
        '            for index in indices:\n'
        '                view = self._fused_views[index]\n'
        '                if not isinstance(view, Image.Image):\n'
        '                    view = Image.fromarray(np.asarray(view))\n'
        '                view = view.convert("RGB")\n'
        '                view.thumbnail((336, 336))\n'
        '                buffer = io.BytesIO()\n'
        '                view.save(buffer, format="JPEG", quality=82, optimize=True)\n'
        '                encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))\n'
        '            response = requests.post(\n'
        '                self.fused_url.rstrip("/") + "/score",\n'
        '                json={"instruction": self.instruction, "images": encoded,\n'
        '                      "initial": [index == 0 for index in indices]},\n'
        '                timeout=120,\n'
        '            )\n'
        '            response.raise_for_status()\n'
        '            fused_result = response.json()\n'
        '            if fused_result.get("status") != "ok":\n'
        '                raise RuntimeError("fused reward service returned non-ok status")\n'
        '            fused_bonus = self.fused_weight * float(fused_result["bonus"])\n'
        '            if not math.isfinite(fused_bonus) or not 0 <= fused_bonus <= self.fused_weight:\n'
        '                raise ValueError("invalid fused reward bonus")\n'
        '        added_reward = semantic_reward + floor_reward + fused_bonus\n')
    text = replace_once(text,
        '        info["reward_components"]["success_floor"] = floor_reward\n',
        '        info["reward_components"]["success_floor"] = floor_reward\n'
        '        info["reward_components"]["fused_bonus"] = fused_bonus\n'
        '        info["fused_reward"] = fused_result\n')
    source[path] = text

    for relative, contents in source.items():
        (root / relative).write_text(contents)
    print("installed checksum-guarded fused reward hook")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_env_patch.py ISOLATED_ACTIVEVLN_ROOT")
    patch(Path(sys.argv[1]))
