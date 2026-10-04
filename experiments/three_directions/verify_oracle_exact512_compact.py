"""Independently recount the published exact512 paired episode exports.

The remote evaluator's raw shard logs are checked separately before export.
This verifier uses only the frozen manifest, compact paired episodes, final
validators, coverage records, and per-seed analysis in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


FULL_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SCREEN_SHA = "bf5ddb4a5c5dd1dbb1c9e1272d91963a9fc76986cb265ac79b8b41c65eaa0a1c"
EPS = 1e-9


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual: float, expected: float, description: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > EPS:
        raise ValueError(f"{description}: {actual} != {expected}")


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("empty evaluation scope")
    output = {"episodes": n}
    for arm in ("candidate", "control"):
        successes = sum(row[arm]["success"] for row in rows)
        output[arm] = {
            "successes": successes,
            "sr": successes / n,
            "spl": sum(row[arm]["spl"] for row in rows) / n,
            "mean_distance_to_goal": sum(
                row[arm]["distance_to_goal_m"] for row in rows) / n,
        }
    output["paired"] = {
        "sr_pp": 100 * (output["candidate"]["sr"] - output["control"]["sr"]),
        "spl_pp": 100 * (output["candidate"]["spl"] - output["control"]["spl"]),
        "candidate_only_successes": sum(
            row["candidate"]["success"] and not row["control"]["success"]
            for row in rows),
        "control_only_successes": sum(
            row["control"]["success"] and not row["candidate"]["success"]
            for row in rows),
    }
    return output


def check_scope(actual: dict, reported: dict, label: str) -> None:
    if actual["episodes"] != reported["episodes"]:
        raise ValueError(f"{label}: episode count")
    for arm in ("candidate", "control"):
        expected = reported[f"{arm}_metrics"]
        if actual[arm]["successes"] != expected["successes"] or \
                expected["count"] != actual["episodes"] or \
                expected["inference_errors"] != 0:
            raise ValueError(f"{label}: {arm} coverage, errors, or successes")
        for name in ("sr", "spl", "mean_distance_to_goal"):
            near(actual[arm][name], expected[name], f"{label}: {arm} {name}")
    for name, value in actual["paired"].items():
        expected = reported["paired"][name]
        if isinstance(value, int):
            if value != expected:
                raise ValueError(f"{label}: {name}")
        else:
            near(value, expected, f"{label}: {name}")


def read_seed(package: Path, seed: int, ids: list[str], scenes: list[str],
              screen_ids: set[str]) -> dict:
    candidate_name = f"oracle_turnwise_exact512_128_seed{seed}"
    control_name = f"oracle_exact512_control_128_seed{seed}"
    stem = f"seed{seed}_full1839"
    export_path = package / f"{stem}_paired_episodes.jsonl"
    lines = export_path.read_text().splitlines()
    if len(lines) != len(ids):
        raise ValueError(f"seed {seed}: export count")
    rows = [json.loads(line) for line in lines]
    if [row["episode_id"] for row in rows] != ids or \
            [row["scene_id"] for row in rows] != scenes:
        raise ValueError(f"seed {seed}: frozen manifest order or scene mismatch")
    for row in rows:
        if set(row) != {"episode_id", "scene_id", "candidate", "control"}:
            raise ValueError("unexpected export columns")
        for arm in ("candidate", "control"):
            value = row[arm]
            if set(value) != {"success", "spl", "distance_to_goal_m",
                              "oracle_success", "early_stop_reason"} or \
                    type(value["success"]) is not bool or \
                    type(value["oracle_success"]) is not bool or \
                    not 0 <= float(value["spl"]) <= 1 or \
                    not math.isfinite(float(value["distance_to_goal_m"])) or \
                    float(value["distance_to_goal_m"]) < 0 or \
                    value["early_stop_reason"] == "inference_error":
                raise ValueError(f"seed {seed}: invalid {arm} episode value")
    coverage = json.loads((package / f"{stem}_coverage.json").read_text())
    if coverage.get("schema") != "exact512_eval_progress_v1" or \
            coverage.get("manifest_sha256") != FULL_SHA or \
            len(coverage.get("labels", [])) != 2:
        raise ValueError(f"seed {seed}: invalid coverage report")
    validators = {}
    for arm, expected_name in (("candidate", candidate_name),
                               ("control", control_name)):
        path = package / f"{arm}_seed{seed}_validated.json"
        validator = json.loads(path.read_text())
        matches = [label for label in coverage["labels"]
                   if label["label"] == expected_name]
        if len(matches) != 1 or validator != {
                "label": expected_name, "episodes": len(ids),
                "successes": matches[0]["validated_successes"],
                "inference_errors": 0} or \
                matches[0]["validator_sha256"] != digest(path) or \
                not matches[0]["completed"] or matches[0]["failed"] or \
                matches[0]["unique_stats"] != len(ids) or \
                matches[0]["missing"] != 0 or \
                matches[0]["per_shard"] != [460, 460, 460, 459]:
            raise ValueError(f"seed {seed}: {arm} validator mismatch")
        validators[arm] = validator
    analysis = json.loads((package / f"{stem}_early_analysis.json").read_text())
    if analysis.get("schema") != "oracle_exact512_one_seed_full_val_unseen_v1" or \
            analysis.get("seed") != seed or \
            analysis.get("candidate") != candidate_name or \
            analysis.get("control") != control_name or \
            analysis.get("full_manifest_sha256") != FULL_SHA or \
            analysis.get("screen_manifest_sha256") != SCREEN_SHA or \
            analysis.get("candidate_validator_sha256") != digest(
                package / f"candidate_seed{seed}_validated.json") or \
            analysis.get("control_validator_sha256") != digest(
                package / f"control_seed{seed}_validated.json"):
        raise ValueError(f"seed {seed}: analysis identity mismatch")
    scopes = {}
    for name, selected in (("full", rows),
                           ("reused_screen", [row for row in rows
                                              if row["episode_id"] in screen_ids]),
                           ("outside_reused_screen", [row for row in rows
                                                      if row["episode_id"] not in screen_ids])):
        result = metrics(selected)
        check_scope(result, analysis[name], f"seed {seed}/{name}")
        scopes[name] = result
    for arm in ("candidate", "control"):
        if scopes["full"][arm]["successes"] != validators[arm]["successes"]:
            raise ValueError(f"seed {seed}: {arm} validator successes mismatch")
    return {"seed": seed, "export_sha256": digest(export_path),
            "analysis_sha256": digest(package / f"{stem}_early_analysis.json"),
            "full": scopes["full"],
            "outside_reused_screen": scopes["outside_reused_screen"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--three-seed-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.full_manifest) != FULL_SHA or \
            digest(args.screen_manifest) != SCREEN_SHA or \
            len(args.seeds) != len(set(args.seeds)) or \
            any(seed not in (11, 22, 33) for seed in args.seeds):
        raise ValueError("manifest, seed, or split mismatch")
    full = json.loads(args.full_manifest.read_text())
    screen = json.loads(args.screen_manifest.read_text())
    ids = [str(eid) for eid in full["episode_ids"]]
    scenes = [str(scene) for scene in full["scene_ids"]]
    screen_ids = {str(eid) for eid in screen["episode_ids"]}
    if len(ids) != len(set(ids)) or len(ids) != len(scenes) or \
            len(ids) != 1839 or len(set(scenes)) != 11 or \
            len(screen_ids) != 256 or not screen_ids.issubset(ids):
        raise ValueError("wrong frozen episode coverage")
    seed_results = [read_seed(args.package, seed, ids, scenes, screen_ids)
                    for seed in args.seeds]
    report = {"schema": "oracle_exact512_compact_independent_recount_v1",
              "full_manifest_sha256": FULL_SHA,
              "screen_manifest_sha256": SCREEN_SHA,
              "seeds": args.seeds, "pairs": seed_results,
              "three_seed": None,
              "interpretation": "Compact-export recount; raw shard logs are validated before export. Privileged reward and reused val-unseen development data."}
    if args.seeds == [11, 22, 33]:
        report["three_seed"] = {
            scope: {name: {
                "per_seed": [row[scope]["paired"][name] for row in seed_results],
                "mean": statistics.mean(row[scope]["paired"][name]
                                        for row in seed_results),
                "sample_sd": statistics.stdev(row[scope]["paired"][name]
                                               for row in seed_results),
            } for name in ("sr_pp", "spl_pp")}
            for scope in ("full", "outside_reused_screen")}
    if args.three_seed_analysis is not None:
        if report["three_seed"] is None:
            raise ValueError("three-seed report requires all paired seeds")
        final = json.loads(args.three_seed_analysis.read_text())
        if final.get("schema") != "oracle_exact512_three_seed_full_val_unseen_v1" or \
                final.get("full_manifest_sha256") != FULL_SHA or \
                final.get("screen_manifest_sha256") != SCREEN_SHA or \
                final.get("seeds") != [11, 22, 33]:
            raise ValueError("three-seed report identity mismatch")
        for scope in ("full", "outside_reused_screen"):
            expected = final[scope]
            count = 1839 if scope == "full" else 1583
            if expected["episodes_per_seed"] != count or \
                    expected["scenes"] != 11 or \
                    [p["seed"] for p in expected["pairs"]] != [11, 22, 33]:
                raise ValueError(f"three-seed {scope} coverage mismatch")
            for index, result in enumerate(seed_results):
                pair = expected["pairs"][index]
                if pair["candidate_metrics"]["successes"] != \
                        result[scope]["candidate"]["successes"] or \
                        pair["control_metrics"]["successes"] != \
                        result[scope]["control"]["successes"]:
                    raise ValueError(f"three-seed {scope} seed successes")
            for name in ("sr_pp", "spl_pp"):
                actual = report["three_seed"][scope][name]
                listed = expected["metrics"][name]
                for got, want in zip(actual["per_seed"], listed["per_seed"]):
                    near(got, want, f"three-seed {scope} {name} per-seed")
                near(actual["mean"], listed["mean"],
                     f"three-seed {scope} {name} mean")
                near(actual["sample_sd"], listed["sample_sd"],
                     f"three-seed {scope} {name} sample SD")
        report["three_seed_analysis_sha256"] = digest(
            args.three_seed_analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seeds": args.seeds,
                      "three_seed": report["three_seed"]}, indent=2))


if __name__ == "__main__":
    main()
