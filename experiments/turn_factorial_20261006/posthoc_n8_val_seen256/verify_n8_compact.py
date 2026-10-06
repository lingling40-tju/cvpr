"""Recount the three-arm n=8 val-seen pilot from compact episode metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
N8 = "norm_terminal_n8_64step_seed11"
COMPARISONS = {
    "n8_vs_control.json": "qwen3_exact_control_64step_seed11",
    "n8_vs_n4.json": "norm_terminal_rloo_64step_seed11",
}


def main() -> None:
    compact = json.loads((HERE / "n8_three_arm_compact.json").read_text())
    rows = compact["rows"]
    assert compact["split"] == "val_seen"
    assert len(rows) == 256
    assert len({str(row["episode_id"]) for row in rows}) == 256
    assert len({row["scene_id"] for row in rows}) == 38
    for row in rows:
        for arm in (*COMPARISONS.values(), N8):
            result = row[arm]
            assert result["success"] in (0, 1, 0.0, 1.0)
            assert math.isfinite(result["spl"]) and 0 <= result["spl"] <= 1

    for filename, control in COMPARISONS.items():
        report = json.loads((HERE / filename).read_text())
        control_successes = sum(row[control]["success"] for row in rows)
        candidate_successes = sum(row[N8]["success"] for row in rows)
        sr_points = 100 * (candidate_successes - control_successes) / len(rows)
        spl_points = 100 * sum(row[N8]["spl"] - row[control]["spl"]
                               for row in rows) / len(rows)
        assert report["control"] == control and report["candidate"] == N8
        assert report["episodes"] == 256 and report["scenes"] == 38
        assert report["control_successes"] == control_successes
        assert report["candidate_successes"] == candidate_successes
        assert math.isclose(report["paired_sr_points"], sr_points, abs_tol=1e-10)
        assert math.isclose(report["paired_spl_points"], spl_points,
                            abs_tol=1e-10)
        print(f"{control}: {int(control_successes)} -> {int(candidate_successes)}, "
              f"SR {sr_points:+.6f} pp, SPL {spl_points:+.6f} pp")


if __name__ == "__main__":
    main()
