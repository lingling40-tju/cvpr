#!/usr/bin/env python3
"""Verify archived code/config hashes against the frozen remote identity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
identity = json.loads((ROOT / "runlogs/freeze/identity.json").read_text())
snapshot = ROOT / "source_snapshot"
omitted = {"prepared_data/fit512.parquet", "prepared_data/fit512_manifest.json"}
checked = []
for rel, expected in identity["source_sha256"].items():
    if rel in omitted:
        continue
    path = (ROOT / rel) if rel == "prepared_data/development256.json" else (snapshot / rel)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA mismatch: {rel}")
    checked.append(rel)
if len(checked) != len(identity["source_sha256"]) - len(omitted):
    raise SystemExit("unexpected source inventory")
print(json.dumps({"status": "PASS", "checked_files": len(checked),
                  "omitted_fit_data": sorted(omitted),
                  "identity_sha256": identity["identity_sha256"]}, indent=2))
