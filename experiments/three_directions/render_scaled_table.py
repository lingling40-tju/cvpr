"""Render a CVPR table only from completed six-model analysis JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (11, 22, 33)


def checked_analysis(path: Path, count: int) -> dict:
    data = json.loads(path.read_text())
    assert data["split"] == "val_unseen"
    assert data["mode"] == "branch" and data["train_steps"] == 128
    assert data["episodes"] == count and data["scenes"] == 11
    assert data["train_rows_per_arm"] == 512 and data["rollouts_per_episode"] == 2
    expected_labels = {
        f"branch128_seed{seed}" for seed in SEEDS
    } | {
        f"branch_control128_seed{seed}" for seed in SEEDS
    }
    assert set(data["models"]) == expected_labels
    assert set(data["paired_seed_differences"]) == {str(seed) for seed in SEEDS}
    for label, model in data["models"].items():
        assert model["count"] == count and model["inference_errors"] == 0, label
        assert 0 <= model["successes"] <= count
        assert abs(model["sr"] - model["successes"] / count) < 1e-12
        assert 0 <= model["spl"] <= 1
    for seed in SEEDS:
        pair = data["paired_seed_differences"][str(seed)]
        candidate_label = f"branch128_seed{seed}"
        control_label = f"branch_control128_seed{seed}"
        assert pair["candidate_label"] == candidate_label
        assert pair["control_label"] == control_label
        candidate = data["models"][candidate_label]
        control = data["models"][control_label]
        assert abs(pair["sr_pp"] - 100 * (candidate["sr"] - control["sr"])) < 1e-9
        assert abs(pair["spl_pp"] - 100 * (candidate["spl"] - control["spl"])) < 1e-9
        assert pair["candidate_only_successes"] - pair["control_only_successes"] == (
            candidate["successes"] - control["successes"]
        )
    for metric in ("sr_pp", "spl_pp"):
        values = [data["paired_seed_differences"][str(seed)][metric] for seed in SEEDS]
        assert abs(data[f"mean_paired_{metric}"] - statistics.mean(values)) < 1e-9
        assert abs(data[f"sd_paired_{metric}"] - statistics.stdev(values)) < 1e-9
    return data


def section(label: str, data: dict) -> list[str]:
    lines = [f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\", "\\midrule"]
    for seed in SEEDS:
        control = data["models"][f"branch_control128_seed{seed}"]
        branch = data["models"][f"branch128_seed{seed}"]
        pair = data["paired_seed_differences"][str(seed)]
        lines.append(
            f"{seed} & {100 * control['sr']:.2f} & {100 * branch['sr']:.2f} "
            f"& {pair['sr_pp']:+.2f} & {100 * control['spl']:.2f} "
            f"& {100 * branch['spl']:.2f} & {pair['spl_pp']:+.2f} \\\\"
        )
    lines.append(
        "Mean $\\pm$ SD & -- & -- & "
        f"${data['mean_paired_sr_pp']:+.2f} \\pm {data['sd_paired_sr_pp']:.2f}$ "
        "& -- & -- & "
        f"${data['mean_paired_spl_pp']:+.2f} \\pm {data['sd_paired_spl_pp']:.2f}$ \\\\"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = checked_analysis(args.screen, 256)
    full = checked_analysis(args.full, 1839)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Matched three-seed R2R val-unseen results. Control (C) starts at the task initial pose; branch (B) replays a training-only policy prefix. SR and SPL are percentages; differences are B minus C in percentage points. Each model is evaluated on every episode in the stated split with zero inference errors.}",
        "\\label{tab:branch-scaled}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Seed & C SR & B SR & $\\Delta$ SR & C SPL & B SPL & $\\Delta$ SPL \\\\",
        "\\midrule",
        *section("Frozen 256-episode screen", screen),
        "\\midrule",
        *section("Complete 1,839-episode val-unseen", full),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
