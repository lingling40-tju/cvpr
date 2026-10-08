"""Guard the additional precision diagnostic without changing its parent freeze."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--identity-sha", required=True)
    a = p.parse_args()
    identity = a.root / "prepared_data/positive_matched_precision_identity.json"
    if sha(identity) != a.identity_sha:
        raise ValueError("precision freeze digest differs from the launch identity")
    f = json.loads(identity.read_text())
    if f["schema"] != "positive_matched_precision_freeze_v1" or \
            f["new_precision_reference_opened_at_freeze"] is not False or \
            f["parent_sft_reference_opened_at_freeze"] is not False or \
            f["sft_cli_dtype"] != "half" or f["required_actual_engine_dtype"] != "float16":
        raise ValueError("precision protocol identity differs")
    for name, digest in f["source_sha256"].items():
        if sha(a.root / name) != digest:
            raise ValueError(f"frozen precision source differs: {name}")
    for name, digest in f["installed_runtime_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"installed runtime differs: {name}")
    parent = json.loads((a.root / "prepared_data/positive_extra_protocol_identity.json").read_text())
    for name, digest in parent["source_sha256"].items():
        if sha(a.root / name) != digest:
            raise ValueError(f"original frozen source differs: {name}")
    if f["generation"] != parent["generation"] or f["sft_independent_training_seeds"] != 0:
        raise ValueError("generation or shared-reference scope differs")
    print(json.dumps({"freeze_sha256": sha(identity), "source_identity_verified": True}))


if __name__ == "__main__":
    main()
