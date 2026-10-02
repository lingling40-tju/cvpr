"""Independently recompute publication metrics from compact paired JSONLs.

This deliberately does not import the raw-evaluation analyzers. It checks the
archived numbers and the screen-disjoint subset from the published records.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "scale_full1839"
SEEDS = (11, 22, 33)


def close(observed: float, expected: float) -> None:
    assert math.isclose(observed, expected, rel_tol=0, abs_tol=1e-10), (observed, expected)


def load_rows(count: int, seed: int) -> dict[str, dict]:
    path = ROOT / f"paired_{count}_seed{seed}.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        episode_id = str(item["episode_id"])
        assert episode_id not in rows
        assert item["scene_id"]
        for arm in ("candidate", "control"):
            stats = item[arm]
            assert type(stats["success"]) is bool
            assert math.isfinite(float(stats["spl"])) and 0 <= float(stats["spl"]) <= 1
            assert math.isfinite(float(stats["distance_to_goal_m"]))
        rows[episode_id] = item
    assert len(rows) == count
    return rows


def summarize(rows: dict[str, dict], arm: str) -> dict:
    n = len(rows)
    return {
        "count": n,
        "successes": sum(item[arm]["success"] for item in rows.values()),
        "sr": sum(item[arm]["success"] for item in rows.values()) / n,
        "spl": sum(float(item[arm]["spl"]) for item in rows.values()) / n,
        "mean_distance_to_goal": sum(
            float(item[arm]["distance_to_goal_m"]) for item in rows.values()
        ) / n,
    }


def difference(rows: dict[str, dict]) -> dict:
    n = len(rows)
    return {
        "sr_pp": 100 * sum(
            item["candidate"]["success"] - item["control"]["success"]
            for item in rows.values()
        ) / n,
        "spl_pp": 100 * sum(
            float(item["candidate"]["spl"]) - float(item["control"]["spl"])
            for item in rows.values()
        ) / n,
        "candidate_only_successes": sum(
            item["candidate"]["success"] and not item["control"]["success"]
            for item in rows.values()
        ),
        "control_only_successes": sum(
            item["control"]["success"] and not item["candidate"]["success"]
            for item in rows.values()
        ),
    }


def check_analysis(analysis: dict, rows_by_seed: dict[int, dict[str, dict]]) -> None:
    n = analysis["episodes"]
    assert set(rows_by_seed) == set(SEEDS)
    assert all(len(rows_by_seed[seed]) == n for seed in SEEDS)
    assert all(
        set(rows_by_seed[seed]) == set(rows_by_seed[11]) for seed in SEEDS
    )
    assert len({item["scene_id"] for item in rows_by_seed[11].values()}) == analysis["scenes"]
    for seed in SEEDS:
        rows = rows_by_seed[seed]
        candidate_label = f"branch128_seed{seed}"
        control_label = f"branch_control128_seed{seed}"
        for arm, label in (("candidate", candidate_label), ("control", control_label)):
            observed = summarize(rows, arm)
            expected = analysis["models"][label]
            assert observed["count"] == expected["count"]
            assert observed["successes"] == expected["successes"]
            assert expected["inference_errors"] == 0
            for key in ("sr", "spl", "mean_distance_to_goal"):
                close(observed[key], expected[key])
        observed_diff = difference(rows)
        expected_diff = analysis["paired_seed_differences"][str(seed)]
        assert expected_diff["candidate_label"] == candidate_label
        assert expected_diff["control_label"] == control_label
        for key, value in observed_diff.items():
            if key.endswith("successes"):
                assert value == expected_diff[key]
            else:
                close(value, expected_diff[key])
    for key in ("sr_pp", "spl_pp"):
        values = [difference(rows_by_seed[seed])[key] for seed in SEEDS]
        close(statistics.mean(values), analysis[f"mean_paired_{key}"])
        close(statistics.stdev(values), analysis[f"sd_paired_{key}"])


def check_hashes() -> None:
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        path = ROOT / filename
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename


def main() -> None:
    check_hashes()
    screen = {seed: load_rows(256, seed) for seed in SEEDS}
    full = {seed: load_rows(1839, seed) for seed in SEEDS}
    screen_ids = set(screen[11])
    assert screen_ids <= set(full[11])
    for seed in SEEDS:
        assert set(screen[seed]) == screen_ids
        for eid in screen_ids:
            assert screen[seed][eid]["scene_id"] == full[seed][eid]["scene_id"]
    holdout = {
        seed: {eid: row for eid, row in full[seed].items() if eid not in screen_ids}
        for seed in SEEDS
    }
    assert all(len(holdout[seed]) == 1583 for seed in SEEDS)
    for name, rows in (
        ("val256_analysis.json", screen),
        ("full1839_analysis.json", full),
        ("holdout1583_analysis.json", holdout),
    ):
        analysis = json.loads((ROOT / name).read_text())
        check_analysis(analysis, rows)
        print(name, "verified", analysis["mean_paired_sr_pp"], analysis["mean_paired_spl_pp"])


if __name__ == "__main__":
    main()
