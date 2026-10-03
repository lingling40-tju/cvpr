"""Summarize three matched full val-unseen reward comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


MANIFEST_SHA256 = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-prefix", default="fused_group4_128",
                        choices=("fused_group4_128", "failure_only_group4_128",
                                 "stopaware_group4_128"))
    args = parser.parse_args()
    actual = hashlib.sha256((args.root / "manifest.json").read_bytes()).hexdigest()
    if actual != MANIFEST_SHA256:
        raise ValueError("full val-unseen manifest hash mismatch")
    pairs = []
    for seed in (11, 22, 33):
        candidate = f"{args.candidate_prefix}_seed{seed}"
        control = f"group4_128_seed{seed}"
        if not (args.root / f"{candidate}.completed").is_file() or not \
                (args.root / f"{control}.completed").is_file():
            raise ValueError(f"incomplete candidate/control seed {seed}")
        pair = json.loads((args.root / f"paired_{candidate}_vs_{control}.json").read_text())
        if pair["split"] != "val_unseen" or pair["episodes"] != 1839 or \
                pair["manifest_sha256"] != actual or \
                pair["candidate"] != candidate or pair["control"] != control:
            raise ValueError(f"invalid matched pair seed {seed}")
        for arm in ("candidate_metrics", "control_metrics"):
            if pair[arm]["count"] != 1839 or pair[arm]["inference_errors"] != 0:
                raise ValueError(f"incomplete or erroneous inference seed {seed}")
        if not all(math.isfinite(pair["paired"][metric])
                   for metric in ("sr_pp", "spl_pp")):
            raise ValueError(f"nonfinite paired metric seed {seed}")
        pairs.append({"seed": seed, "candidate": candidate, "control": control,
                      "candidate_metrics": pair["candidate_metrics"],
                      "control_metrics": pair["control_metrics"],
                      "paired": pair["paired"]})
    summary = {}
    for metric in ("sr_pp", "spl_pp"):
        values = [pair["paired"][metric] for pair in pairs]
        summary[metric] = {"values": values, "mean": statistics.mean(values),
                           "sample_sd": statistics.stdev(values)}
    result = {"mode": args.candidate_prefix, "split": "val_unseen",
              "episodes_per_seed": 1839, "manifest_sha256": actual,
              "seeds": [11, 22, 33], "pairs": pairs, "summary": summary,
              "interpretation": "Three-seed full validation comparison against matched group-four outcome-only controls. Descriptive mean and SD, not a confidence interval."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
