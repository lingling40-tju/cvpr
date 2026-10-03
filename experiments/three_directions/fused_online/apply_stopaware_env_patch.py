"""Create a stop-conditioned, centered terminal potential in an isolated tree.

The source failure-only experiment is complete. This patch accepts only its
exact environment wrapper and must be applied to a separate code checkout.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


EXPECTED_SHA256 = "04ab342e477c1353d46446a0e64f51894ce7ac263607f01967719707defd27f9"


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"expected one patch location: {before[:60]!r}")
    return source.replace(before, after)


def main(root: Path) -> None:
    path = root / "vlnce_server/semantic_reward/env.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("isolated environment wrapper checksum mismatch")
    source = raw.decode()
    source = replace_once(
        source,
        '        fused_result = {"status": "disabled"}\n',
        '        fused_result = {"status": "censored" if done and not info.get("task_success") else "disabled",\n'
        '                        "reason": info.get("end_reason") if done else None}\n',
    )
    source = replace_once(
        source,
        '        if done and self.fused_weight > 0 and not info.get("task_success"):\n',
        '        if (done and self.fused_weight > 0 and not info.get("task_success")\n'
        '                and info.get("end_reason") == "stopped but goal not reached."):\n',
    )
    source = replace_once(
        source,
        '            fused_bonus = self.fused_weight * float(fused_result["bonus"])\n',
        '            raw_bonus = float(fused_result["bonus"])\n'
        '            if not math.isfinite(raw_bonus) or not 0 <= raw_bonus <= 1:\n'
        '                raise ValueError("invalid raw potential bonus")\n'
        '            fused_bonus = self.fused_weight * max(0.0, 2.0 * raw_bonus - 1.0)\n'
        '            fused_result["applied_bonus"] = fused_bonus\n',
    )
    path.write_text(source)
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_stopaware_env_patch.py ISOLATED_ROOT")
    main(Path(sys.argv[1]))
