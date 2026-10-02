"""Switch the isolated fused hook to failure-only requests after its screen.

This must only be applied once the original frozen-fusion 256-episode screen
has ruled out its three-seed scale suite.  The source experiment tree is
untouched.  The checksum prevents silently changing a different wrapper.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


EXPECTED_SHA256 = "8051105a590ca9232b145d9ebbbb017c96ad8983877e24437a41875393991298"


def main(root: Path) -> None:
    path = root / "vlnce_server/semantic_reward/env.py"
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA256:
        raise ValueError("isolated fused environment wrapper changed")
    before = "        if done and self.fused_weight > 0:\n"
    after = ("        if done and self.fused_weight > 0 and "
             "not info.get(\"task_success\"):\n")
    text = original.decode()
    if text.count(before) != 1:
        raise ValueError("expected one terminal fused-reward hook")
    text = text.replace(before, after)
    path.write_text(text)
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_failure_only_env_patch.py ISOLATED_ROOT")
    main(Path(sys.argv[1]))
