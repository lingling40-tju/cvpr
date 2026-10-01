"""Check full val-unseen coverage and shard summaries before marking a model done."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("shards", type=int)
    args = parser.parse_args()
    expected = [str(item) for item in json.loads(args.manifest.read_text())["episode_ids"]]
    observed = []
    successes = 0
    errors = 0
    for shard in range(args.shards):
        folder = args.result_root / args.label / f"shard_{shard:02d}"
        summary = json.loads((folder / "summary.json").read_text())
        ids = [str(item) for item in summary["episode_ids"]]
        assert summary["count"] == len(ids) == len(expected[shard::args.shards])
        assert len(set(ids)) == len(ids)
        assert set(ids) == set(expected[shard::args.shards])
        assert len(list((folder / "log").glob("stats_*_0.json"))) == len(ids)
        observed.extend(ids)
        successes += summary["successes"]
        errors += summary["inference_errors"]
    assert len(observed) == len(expected) and set(observed) == set(expected)
    assert errors == 0, f"{errors} inference errors for {args.label}"
    print(json.dumps({"label": args.label, "episodes": len(observed),
                      "successes": successes, "inference_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
