"""Independent arithmetic recount of recorded trajectory-credit diagnostics."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def weights(scores, success, priority=15, floor=2.5, cap=1.5):
    quality = [r + priority * s for r, s in zip(scores, success)]
    return [min(cap, max(0, quality[i] - sum(quality[j] for j in range(4) if j != i) / 3) / 5)
            if success[i] or scores[i] >= floor else 0 for i in range(4)]


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "Output already exists")
    r = json.loads(a.report.read_text())
    seed = r["configured_seed"]
    audit_path = a.root / f"scale128/candidate_seed{seed}_train_audit.json"
    tb_path = a.root / f"scale128/candidate_seed{seed}_tensorboard.json"
    budget = json.loads((a.root / f"scale128/candidate_seed{seed}_rollout_budget.json").read_text())
    tb = json.loads(tb_path.read_text()); scalar = {x["step"]: x for x in tb["steps"]}
    fit_path = a.root / "prepared_data/fit512_manifest.json"; fit = set(json.loads(fit_path.read_text())["episode_ids"])
    require(r["schema"] == "positive_recorded_credit_mechanism_v1" and
            r["run"] == f"positive_trajectory_candidate_128step_seed{seed}" and
            r["compact_sha256"] == sha(a.compact) and r["training_audit_sha256"] == sha(audit_path) and
            r["tensorboard_export_sha256"] == sha(tb_path) and r["fit_manifest_sha256"] == sha(fit_path) and
            r["rollout_sha256"] == budget["rollout_sha256"] and
            r["source_sha256"]["verl/trainer/ppo/positive_trajectory_advantage.py"] ==
            sha(a.root / "verl/trainer/ppo/positive_trajectory_advantage.py"), "Evidence identity differs")
    require(r["constants"] == {"GROUP_SIZE": 4, "SUCCESS_PRIORITY": 15.0, "FAILURE_SCORE_FLOOR": 2.5,
                               "GAP_SCALE": 5.0, "MAX_ADVANTAGE": 1.5}, "Credit constants differ")
    counts, changes, sums, by_success = Counter(), Counter(), Counter(), Counter()
    steps = defaultdict(list)
    max_formula_error = 0.0
    for line in a.compact.open():
        row = json.loads(line)
        step, episode = row["step"], row["episode_id"]
        require(type(step) is int and 1 <= step <= 128 and episode in fit, "Record identity differs")
        scores, success, actual = row["score_float32"], row["task_success"], row["reconstructed_weights"]
        require(len(scores) == len(success) == len(actual) == len(row["recorded_terminal_scores"]) == 4,
                "Group size differs")
        require(all(math.isfinite(x) and 0 <= x <= 20 for x in scores) and
                all(x in (0, 1) for x in success) and
                all(f32(v) == x for v, x in zip(row["recorded_terminal_scores"], scores)), "Score inputs differ")
        require(all(math.isfinite(w) and 0 <= w <= 1.5 for w in actual), "Credit range differs")
        variants = row["same_record_alternative_weights"]
        expected = {"no_success_priority": weights(scores, success, priority=0),
                    "no_failure_floor": weights(scores, success, floor=0),
                    "no_cap": weights(scores, success, cap=float("inf")),
                    "constant_success_proxy": [1.5 * s for s in success]}
        require(set(variants) == set(expected), "Alternative definitions differ")
        for got, target in [(actual, weights(scores, success))] + [(variants[k], v) for k, v in expected.items()]:
            require(len(got) == 4, "Alternative group size differs")
            error = max(abs(x - y) for x, y in zip(got, target)); max_formula_error = max(error, max_formula_error)
            require(error <= 1e-5, "Independent credit arithmetic differs")
        n_success = sum(success); by_success[str(n_success)] += 1; counts["groups"] += 1
        counts["all_failure_groups_with_positive_credit"] += n_success == 0 and any(w > 0 for w in actual)
        for i, w in enumerate(actual):
            s = success[i]; counts["trajectories"] += 1; counts["successful_trajectories"] += s
            counts["failed_positive_score"] += not s and row["recorded_terminal_scores"][i] > 0
            counts["failed_floor_eligible"] += not s and scores[i] >= 2.5
            counts["positive_credit_trajectories"] += w > 0
            counts["positive_credit_successes"] += bool(s and w > 0)
            counts["positive_credit_failures"] += bool(not s and w > 0)
            counts["successful_zero_credit"] += bool(s and w == 0)
            counts["capped_positive_credit"] += w == 1.5
            counts["uncapped_positive_credit"] += 0 < w < 1.5
            for key, values in variants.items():
                difference = abs(w - values[i]); changes[key] += difference > 1e-5; sums[key] += difference
        steps[step].append(row)
    require(sorted(steps) == list(range(1, 129)), "Step coverage differs")
    for step, groups in steps.items():
        require(len(groups) == len({g["episode_id"] for g in groups}) == 8 and
                sorted(g["episode_id"] for g in groups) == budget["steps"][step - 1]["episode_ids"],
                "Episode membership differs from independent budget export")
        scores = [v for g in groups for v in g["score_float32"]]
        actual = [v for g in groups for v in g["reconstructed_weights"]]
        success = [v for g in groups for v in g["task_success"]]
        for key, value in {"critic/score/mean": sum(scores) / 32, "critic/score/max": max(scores),
                           "critic/advantages/min": 0, "critic/advantages/max": max(actual),
                           "agent_reason/[successfully reached the goal.]": sum(success) / 32}.items():
            require(abs(value - scalar[step][key]) <= 1e-5, "Stored scalar differs")
    require(dict(counts) == r["counts"] and dict(by_success) == r["groups_by_number_of_successes"] and
            dict(changes) == r["same_record_changed_weight_counts"], "Descriptive counts differ")
    require(all(math.isclose(sums[k], r["same_record_absolute_weight_change_sums"][k], rel_tol=0, abs_tol=1e-8)
                for k in sums), "Weight-change totals differ")
    out = {"schema": "positive_credit_mechanism_independent_recount_v1", "run": r["run"],
           "report_sha256": sha(a.report), "compact_sha256": sha(a.compact),
           "verifier_sha256": sha(Path(__file__)), "groups": counts["groups"],
           "trajectories": counts["trajectories"], "maximum_float32_vs_python_formula_error": max_formula_error,
           "counts_and_recorded_scalar_agreement": True,
           "scope": "Independent descriptive reconstruction on fixed recorded trajectories; no policy or navigation ablation."}
    with a.output.open("x") as handle: handle.write(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
