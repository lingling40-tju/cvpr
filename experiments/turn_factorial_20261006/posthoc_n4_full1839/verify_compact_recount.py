"""Independently recount paired full val-unseen metrics for three seeds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent


def main() -> None:
    report = json.loads((HERE / "independent_three_seed_recount.json").read_text())
    assert report["seeds"] == [11, 22, 33]
    assert report["episodes_per_seed"] == 1839
    assert report["scenes"] == 11 and report["inference_errors"] == 0
    first_ids = None
    sr_differences = []
    spl_differences = []
    for pair in report["pairs"]:
        seed = pair["seed"]
        path = HERE / f"seed{seed}_paired_episodes.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pair["compact_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids = {str(row["episode_id"]) for row in rows}
        assert len(rows) == len(ids) == 1839
        assert len({row["scene_id"] for row in rows}) == 11
        if first_ids is None:
            first_ids = ids
        else:
            assert ids == first_ids
        for row in rows:
            for arm in ("candidate", "control"):
                result = row[arm]
                assert result["success"] in (0, 1, 0.0, 1.0)
                assert math.isfinite(result["spl"]) and 0 <= result["spl"] <= 1
        successes = {
            arm: sum(row[arm]["success"] for row in rows)
            for arm in ("candidate", "control")
        }
        sr_points = 100 * (successes["candidate"] - successes["control"]) / 1839
        spl_points = 100 * sum(row["candidate"]["spl"] - row["control"]["spl"]
                               for row in rows) / 1839
        validator = json.loads((HERE / f"norm_terminal_posthoc512_128_seed{seed}.validated.json").read_text())
        assert validator["episodes"] == 1839 and validator["inference_errors"] == 0
        assert validator["successes"] == successes["candidate"]
        assert pair["candidate_successes"] == successes["candidate"]
        assert pair["control_successes"] == successes["control"]
        assert math.isclose(pair["paired_sr_points"], sr_points, abs_tol=1e-10)
        assert math.isclose(pair["paired_spl_points"], spl_points, abs_tol=1e-10)
        sr_differences.append(sr_points)
        spl_differences.append(spl_points)
        print(f"seed {seed}: {int(successes['control'])} -> "
              f"{int(successes['candidate'])}; SR {sr_points:+.6f} pp, "
              f"SPL {spl_points:+.6f} pp")
    for key, values in (("sr", sr_differences), ("spl", spl_differences)):
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        assert math.isclose(report[f"mean_paired_{key}_points"], mean,
                            abs_tol=1e-10)
        assert math.isclose(report[f"sample_sd_paired_{key}_points"], sd,
                            abs_tol=1e-10)
        print(f"{key.upper()} mean {mean:+.6f} pp; seed SD {sd:.6f} pp")


if __name__ == "__main__":
    main()
