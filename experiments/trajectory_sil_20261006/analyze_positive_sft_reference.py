"""Report fixed SFT reference comparisons without inventing SFT training seeds."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from validate_positive_extra import load_metrics, screen, sha, SCREENS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=SCREENS, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--sft-root", type=Path, required=True)
    p.add_argument("--trained-root", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    m, ids = screen(args.manifest, args.role)
    seeds = (11,) if args.role == "development" else (11, 22, 33)
    steps = 64 if args.role == "development" else 128
    sft_label = "positive_initial_sft"
    sft = load_metrics(args.sft_root, sft_label, ids, m["split"])
    labels = [f"positive_trajectory_{arm}_{steps}step_seed{seed}"
              for seed in seeds for arm in ("control", "candidate")]
    metrics = {sft_label: sft}
    metrics.update({label: load_metrics(args.trained_root, label, ids, m["split"])
                    for label in labels})
    compact = [{"episode_id": i, "scene_id": scene_id,
                "models": {label: rows[i] for label, rows in metrics.items()}}
               for i, scene_id in zip(ids, m["scene_ids"])]
    args.compact.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in compact))
    absolute = {label: {"successes": sum(r["success"] for r in rows.values()),
                        "sr": statistics.mean(r["success"] for r in rows.values()),
                        "spl": statistics.mean(r["spl"] for r in rows.values())}
                for label, rows in metrics.items()}
    comparisons = {}
    for label in labels:
        cluster = defaultdict(lambda: [0, 0, 0.0])
        only_trained = only_sft = 0
        for i, scene_id in zip(ids, m["scene_ids"]):
            ds = metrics[label][i]["success"] - sft[i]["success"]
            dp = metrics[label][i]["spl"] - sft[i]["spl"]
            c = cluster[scene_id]
            c[0] += 1; c[1] += ds; c[2] += dp
            only_trained += ds == 1; only_sft += ds == -1
        rng = random.Random(202610071839)
        names = sorted(cluster)
        draws = [[], []]
        for _ in range(10000):
            selected = [cluster[x] for x in rng.choices(names, k=len(names))]
            n = sum(x[0] for x in selected)
            for j in (0, 1):
                draws[j].append(100 * sum(x[j + 1] for x in selected) / n)
        comparisons[label] = {
            "paired_sr_points": 100 * (absolute[label]["sr"] - absolute[sft_label]["sr"]),
            "paired_spl_points": 100 * (absolute[label]["spl"] - absolute[sft_label]["spl"]),
            "trained_only_success": only_trained, "sft_only_success": only_sft,
            "scene_bootstrap_95pct_sr_points": [sorted(draws[0])[249], sorted(draws[0])[9749]],
            "scene_bootstrap_95pct_spl_points": [sorted(draws[1])[249], sorted(draws[1])[9749]],
        }
    out = {"schema": "positive_fixed_sft_reference_v1", "role": args.role,
           "episodes_per_model": len(ids), "scenes": len(set(m["scene_ids"])),
           "manifest_sha256": sha(args.manifest), "compact_sha256": sha(args.compact),
           "inference_errors": 0, "sft_decode_seed": 11, "sft_independent_training_seeds": 0,
           "absolute_metrics": absolute, "comparisons_to_fixed_sft": comparisons,
           "interpretation": "One shared unchanged SFT reference; exploratory comparisons after pilot inspection. No clean held-out claim or SFT seed replication."}
    if len(seeds) == 3:
        out["trained_seed_summary_relative_to_fixed_sft"] = {
            arm: {metric: {"mean_points": statistics.mean(
                    comparisons[f"positive_trajectory_{arm}_128step_seed{s}"][f"paired_{metric}_points"] for s in seeds),
                "sample_sd_points": statistics.stdev(
                    comparisons[f"positive_trajectory_{arm}_128step_seed{s}"][f"paired_{metric}_points"] for s in seeds)}
                  for metric in ("sr", "spl")} for arm in ("control", "candidate")}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"role": args.role, "episodes": len(ids), "models": len(metrics)}))


if __name__ == "__main__":
    main()
