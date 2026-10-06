"""Compare the frozen normalized-screen decision to its independent raw recount."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST = "39fdf160ee4abb6950009be002fa3e7af03f0d31995af61f8e2379af1a831f7b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("normalized_result", type=Path)
    parser.add_argument("independent_recount", type=Path)
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    if not (args.normalized_result / "suite.completed").exists():
        raise ValueError("normalized suite incomplete")
    if (args.normalized_result / "suite.failed").exists():
        raise ValueError("normalized suite marked failed")
    manifest = args.normalized_result / "manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError("normalized manifest changed")
    original = json.loads((args.normalized_result / "conditional_decision.json").read_text())
    recount = json.loads(args.independent_recount.read_text())
    if original["manifest_sha256"] != EXPECTED_MANIFEST or recount["manifest_sha256"] != EXPECTED_MANIFEST:
        raise ValueError("unexpected manifest")
    if original["episodes"] != 256 or recount["episodes"] != 256 or original["scenes"] != 38 or recount["scenes"] != 38:
        raise ValueError("unexpected episode or scene coverage")
    if recount["inference_errors"] != 0:
        raise ValueError("inference errors")
    for measure in ("paired_sr_points", "paired_spl_points"):
        if abs(original[measure] - recount[measure]) >= 1e-6:
            raise ValueError(f"paired result disagreement: {measure}")
    if original["advance_gate_passed"] != recount["advance_gate_passed"]:
        raise ValueError("advancement gate disagreement")
    args.state.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "gae_conditional_normalized_gate_v1",
        "normalized_manifest_sha256": EXPECTED_MANIFEST,
        "independent_recount_sha256": hashlib.sha256(args.independent_recount.read_bytes()).hexdigest(),
        "paired_sr_points": recount["paired_sr_points"],
        "paired_spl_points": recount["paired_spl_points"],
        "advance_gate_passed": recount["advance_gate_passed"],
        "decision": "skip_gae_for_scale" if recount["advance_gate_passed"] else "run_gae_smoke",
    }
    (args.state / "normalized_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    marker = "skipped_for_scale.completed" if recount["advance_gate_passed"] else "normalized_failed_gate.completed"
    (args.state / marker).write_text("verified\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
