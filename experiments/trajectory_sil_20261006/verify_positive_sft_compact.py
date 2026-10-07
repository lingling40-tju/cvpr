"""Recount exported SFT comparisons independently of the raw-data analyzer.

This reads only compact episode metrics, frozen manifests and validation
records. It cannot replace the raw coverage/identity checks performed before
export or claim that a shared SFT decode is independent seed replication.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


SCREENS = {
    "development": ("8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3", 256, 8, "train"),
    "reserved": ("412b3ff0750b4228c530af5b1c28b19f4f56147820af487e956f0539a8b39241", 256, 8, "train"),
    "val_unseen": ("262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e", 1839, 11, "val_unseen"),
}
FREEZE_SHA = "ea382ae88f36cfb069b11e3546e6b1de47254d130069a4acc496bdf321a94c4a"
SFT_SHA = "ea2a5cb50422f1ef3967215d3b55ac9fffe7e233d111d51693a2b7781a44fe47"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual, expected):
    if not isinstance(actual, (int, float)) or isinstance(actual, bool) or \
            not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"metric mismatch: {actual} != {expected}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=SCREENS, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sft-validators", type=Path, required=True)
    p.add_argument("--trained-validators", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--sft-identity", type=Path, required=True)
    p.add_argument("--full-three-seed-report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    digest, count, scene_count, split = SCREENS[args.role]
    if sha(args.manifest) != digest or sha(args.freeze) != FREEZE_SHA or sha(args.sft_identity) != SFT_SHA:
        raise ValueError("frozen input identity differs")
    freeze = json.loads(args.freeze.read_text())
    identity = json.loads(args.sft_identity.read_text())
    if freeze["reserved_opened_at_freeze"] is not False or \
            freeze["all_six_128step_models_evaluated_regardless_reserved_metrics"] is not True or \
            freeze["source_sha256"]["prepared_data/positive_initial_sft_identity.json"] != SFT_SHA or \
            identity["rl_optimizer_steps"] != 0:
        raise ValueError("frozen study provenance differs")
    m = json.loads(args.manifest.read_text())
    ids = [str(i) for i in m["episode_ids"]]
    if m["split"] != split or len(ids) != len(set(ids)) or len(ids) != count or \
            len(m["scene_ids"]) != count or len(set(m["scene_ids"])) != scene_count or \
            (args.role != "val_unseen" and m["role"] != args.role):
        raise ValueError("frozen screen coverage differs")
    report = json.loads(args.report.read_text())
    if report["schema"] != "positive_fixed_sft_reference_v1" or report["role"] != args.role or \
            report["manifest_sha256"] != digest or report["compact_sha256"] != sha(args.compact) or \
            report["episodes_per_model"] != count or report["scenes"] != scene_count or \
            report["inference_errors"] != 0 or report["sft_decode_seed"] != 11 or \
            report["sft_independent_training_seeds"] != 0:
        raise ValueError("report provenance differs")
    seeds = (11,) if args.role == "development" else (11, 22, 33)
    steps = 64 if args.role == "development" else 128
    baseline = "positive_initial_sft"
    labels = [f"positive_trajectory_{arm}_{steps}step_seed{s}"
              for s in seeds for arm in ("control", "candidate")]
    all_labels = [baseline] + labels
    rows = [json.loads(line) for line in args.compact.read_text().splitlines()]
    if [str(r["episode_id"]) for r in rows] != ids or \
            [r["scene_id"] for r in rows] != m["scene_ids"]:
        raise ValueError("compact episode or scene mapping differs")
    totals = {label: [0, 0.0] for label in all_labels}
    for row in rows:
        if set(row["models"]) != set(all_labels):
            raise ValueError("compact model identities differ")
        for label in all_labels:
            result = row["models"][label]
            success, spl = result["success"], result["spl"]
            if type(success) is not int or success not in (0, 1) or isinstance(spl, bool) or \
                    not isinstance(spl, (int, float)) or not math.isfinite(spl) or not 0 <= spl <= 1 or \
                    result["early_stop_reason"] == "inference_error":
                raise ValueError(f"invalid compact navigation metric: {label}")
            for key in ("path_length", "distance_to_goal"):
                value = result[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                        not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid compact {key}: {label}")
            totals[label][0] += success
            totals[label][1] += spl
    if set(report["absolute_metrics"]) != set(all_labels) or \
            set(report["comparisons_to_fixed_sft"]) != set(labels):
        raise ValueError("report model coverage differs")
    for label, (successes, spl_sum) in totals.items():
        root = args.sft_validators if label == baseline else args.trained_validators
        validator = json.loads((root / f"{label}.validated.json").read_text())
        if validator["label"] != label or validator["episodes"] != count or \
                validator["inference_errors"] != 0 or validator["manifest_sha256"] != digest:
            raise ValueError(f"validator identity differs: {label}")
        equal(validator["successes"], successes)
        equal(validator["spl"], spl_sum / count)
        for key, value in (("successes", successes), ("sr", successes / count), ("spl", spl_sum / count)):
            equal(report["absolute_metrics"][label][key], value)
    differences = {}
    for label in labels:
        clusters = defaultdict(lambda: [0, 0, 0.0])
        discordance = [0, 0]
        for row in rows:
            trained, sft = row["models"][label], row["models"][baseline]
            ds, dp = trained["success"] - sft["success"], trained["spl"] - sft["spl"]
            bucket = clusters[row["scene_id"]]
            bucket[0] += 1; bucket[1] += ds; bucket[2] += dp
            discordance[0] += ds == 1; discordance[1] += ds == -1
        sr = 100 * (totals[label][0] - totals[baseline][0]) / count
        spl = 100 * (totals[label][1] - totals[baseline][1]) / count
        comparison = report["comparisons_to_fixed_sft"][label]
        equal(comparison["paired_sr_points"], sr)
        equal(comparison["paired_spl_points"], spl)
        if discordance != [comparison["trained_only_success"], comparison["sft_only_success"]]:
            raise ValueError("paired success discordance differs")
        rng = random.Random(202610071839)
        names = sorted(clusters)
        draws = [[], []]
        for _ in range(10000):
            buckets = [clusters[x] for x in rng.choices(names, k=scene_count)]
            denominator = sum(x[0] for x in buckets)
            for j in (0, 1):
                draws[j].append(100 * sum(x[j + 1] for x in buckets) / denominator)
        for j, metric in enumerate(("sr", "spl")):
            ordered = sorted(draws[j])
            claimed = comparison[f"scene_bootstrap_95pct_{metric}_points"]
            if len(claimed) != 2:
                raise ValueError("scene interval shape differs")
            equal(claimed[0], ordered[249]); equal(claimed[1], ordered[9749])
        differences[label] = {"paired_sr_points": sr, "paired_spl_points": spl}
    if len(seeds) == 3:
        for arm in ("control", "candidate"):
            for metric in ("sr", "spl"):
                values = [differences[f"positive_trajectory_{arm}_128step_seed{s}"][f"paired_{metric}_points"]
                          for s in seeds]
                summary = report["trained_seed_summary_relative_to_fixed_sft"][arm][metric]
                equal(summary["mean_points"], statistics.mean(values))
                equal(summary["sample_sd_points"], statistics.stdev(values))
    if args.full_three_seed_report:
        if args.role != "val_unseen":
            raise ValueError("full three-seed cross-check requires val-unseen")
        raw_report = json.loads(args.full_three_seed_report.read_text())
        if raw_report["manifest_sha256"] != digest or raw_report["seeds"] != [11, 22, 33] or \
                raw_report["schema"] != "turn_credit_reward_three_seed_full1839_independent_recount_v1" or \
                raw_report["episodes_per_seed"] != count or raw_report["scenes"] != scene_count or \
                raw_report["inference_errors"] != 0 or len(raw_report["pairs"]) != 3:
            raise ValueError("full raw report coverage differs")
        by_seed = {pair["seed"]: pair for pair in raw_report["pairs"]}
        if set(by_seed) != set(seeds):
            raise ValueError("full raw report training seeds differ")
        full_deltas = {"sr": [], "spl": []}
        full_clusters = {}
        for seed in seeds:
            pair = by_seed[seed]
            path = args.full_three_seed_report.parent / f"seed{seed}_paired_episodes.jsonl"
            if pair["compact_sha256"] != sha(path):
                raise ValueError("full paired compact digest differs")
            paired_rows = [json.loads(line) for line in path.read_text().splitlines()]
            if [str(r["episode_id"]) for r in paired_rows] != ids or \
                    [r["scene_id"] for r in paired_rows] != m["scene_ids"]:
                raise ValueError("full paired compact mapping differs")
            for arm in ("control", "candidate"):
                label = f"positive_trajectory_{arm}_128step_seed{seed}"
                if pair[arm]["label"] != label or pair[arm]["validator_sha256"] != \
                        sha(args.trained_validators / f"{label}.validated.json"):
                    raise ValueError("full report arm label differs")
                equal(pair[f"{arm}_successes"], totals[label][0])
                equal(pair[f"{arm}_sr"], totals[label][0] / count)
                equal(pair[f"{arm}_spl"], totals[label][1] / count)
                for original, with_sft in zip(paired_rows, rows):
                    for key in ("success", "spl"):
                        value = original[arm][key]
                        if key == "success" and isinstance(value, bool):
                            value = int(value)
                        equal(value, with_sft["models"][label][key])
            clusters = defaultdict(lambda: [0, 0, 0.0])
            discordance = [0, 0]
            for row in rows:
                candidate = row["models"][f"positive_trajectory_candidate_128step_seed{seed}"]
                control = row["models"][f"positive_trajectory_control_128step_seed{seed}"]
                ds, dp = candidate["success"] - control["success"], candidate["spl"] - control["spl"]
                bucket = clusters[row["scene_id"]]
                bucket[0] += 1; bucket[1] += ds; bucket[2] += dp
                discordance[0] += ds == 1; discordance[1] += ds == -1
            if discordance != [pair["candidate_only_success"], pair["control_only_success"]]:
                raise ValueError("full paired success discordance differs")
            full_clusters[seed] = clusters
            for metric in ("sr", "spl"):
                index = 0 if metric == "sr" else 1
                delta = 100 * (totals[f"positive_trajectory_candidate_128step_seed{seed}"][index] -
                               totals[f"positive_trajectory_control_128step_seed{seed}"][index]) / count
                equal(pair[f"paired_{metric}_points"], delta)
                full_deltas[metric].append(delta)
        for metric, values in full_deltas.items():
            equal(raw_report[f"mean_paired_{metric}_points"], statistics.mean(values))
            equal(raw_report[f"sample_sd_paired_{metric}_points"], statistics.stdev(values))
        rng = random.Random(20261006128)
        draws = [[], []]
        for _ in range(10000):
            selected_seeds = rng.choices(seeds, k=3)
            seed_means = [[], []]
            for seed in selected_seeds:
                clusters = full_clusters[seed]
                names = sorted(clusters)
                buckets = [clusters[x] for x in rng.choices(names, k=scene_count)]
                n = sum(x[0] for x in buckets)
                for j in (0, 1):
                    seed_means[j].append(100 * sum(x[j + 1] for x in buckets) / n)
            for j in (0, 1):
                draws[j].append(statistics.mean(seed_means[j]))
        for j, metric in enumerate(("sr", "spl")):
            ordered = sorted(draws[j])
            claimed = raw_report[f"seed_scene_bootstrap_95pct_{metric}_points"]
            if len(claimed) != 2:
                raise ValueError("full seed/scene interval shape differs")
            equal(claimed[0], ordered[249]); equal(claimed[1], ordered[9749])
    out = {"schema": "positive_sft_compact_independent_recount_v1", "role": args.role,
           "episodes_per_model": count, "scenes": scene_count, "models": len(all_labels),
           "compact_sha256": sha(args.compact), "inference_errors": 0,
           "sft_successes": totals[baseline][0], "sft_sr": totals[baseline][0] / count,
           "sft_spl": totals[baseline][1] / count, "comparisons_to_fixed_sft": differences,
           "scene_intervals_recount_agree": True,
           "full_three_seed_exports_agree": bool(args.full_three_seed_report),
           "full_seed_scene_intervals_recount_agree": bool(args.full_three_seed_report),
           "sft_independent_training_seeds": 0,
           "interpretation": "Compact arithmetic/provenance check of exploratory comparisons; not an independent SFT training run or clean confirmatory test."}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
