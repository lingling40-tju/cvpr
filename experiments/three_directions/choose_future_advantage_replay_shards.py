"""Choose sparse Habitat replay concurrency from measured GPU-1 smoke use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose(resource: dict) -> int:
    if resource.get("schema") != "future_advantage_sparse_smoke_resource_v1" or \
            resource.get("wall_seconds", 0) <= 0 or \
            resource.get("samples", 0) <= 0:
        raise ValueError("missing real sparse replay smoke measurement")
    baseline = int(resource["baseline_used_mib"])
    peak = int(resource["peak_used_mib"])
    total = int(resource["gpu_total_mib"])
    if not 0 <= baseline <= peak <= total:
        raise ValueError("impossible GPU memory measurement")
    incremental = max(1500, peak - baseline)
    for count in (4, 2, 1):
        if baseline + count * incremental < .7 * total:
            return count
    raise ValueError("smoke leaves insufficient GPU-1 replay memory margin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", type=Path, required=True)
    args = parser.parse_args()
    print(choose(json.loads(args.resource.read_text())))


if __name__ == "__main__":
    main()
