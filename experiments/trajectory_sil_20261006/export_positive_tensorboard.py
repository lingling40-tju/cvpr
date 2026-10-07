#!/usr/bin/env python3
"""Export completed scale training scalars without running model inference."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

TAGS = (
    "actor/grad_norm", "actor/kl_loss", "actor/entropy",
    "critic/score/mean", "critic/score/max",
    "critic/advantages/min", "critic/advantages/max",
    "agent_reward/success_reward_mean", "agent_reward/ndtw_reward_mean",
    "agent_reward/semantic_reward_mean", "agent_reward/success_floor_mean",
    "agent_reason/[successfully reached the goal.]", "response_length/mean",
    "timing_s/step", "timing_s/gen", "timing_s/update_actor",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("control", "candidate"))
    parser.add_argument("--seed", required=True, type=int, choices=(11, 22, 33))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    name = "positive_trajectory_{}_128step_seed{}".format(args.arm, args.seed)
    run = args.root / "runlogs" / name
    assert (run / "completed").is_file(), "Training has not completed"
    assert not (run / "failed").exists(), "Training failed"
    audit_path = args.root / "runlogs/positive_scale" / (
        "{}_seed{}_train_audit.json".format(args.arm, args.seed))
    audit = json.loads(audit_path.read_text())
    log_path = run / "train.log"
    assert audit["schema"] == "positive_trajectory_training_audit_v1"
    assert audit["arm"] == args.arm
    assert audit["expected_steps"] == audit["observed_steps"] == 128
    assert audit["train_log_sha256"] == sha256(log_path)
    config_path = run / "config.txt"
    config_text = config_path.read_text().strip()
    config_fields = dict(item.split("=", 1) for item in config_text.split())
    fixed_config = {
        "arm": args.arm, "seed": str(args.seed), "steps": "128",
        "rows": "512", "batch_rows": "8", "group_n": "4",
        "dataset_sha256": "240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e",
        "reward": "weighted_success15_plus_ndtw5",
        "loss_agg": "seq-mean-token-mean", "kl_loss_coef": "0.01",
    }
    assert all(config_fields.get(k) == v for k, v in fixed_config.items())

    console = {}
    for line in log_path.read_text(errors="replace").splitlines():
        hit = re.search(r"\bstep:(\d+) - global_seqlen", line)
        if hit:
            step = int(hit.group(1))
            assert step not in console, "Duplicate console optimizer step"
            console[step] = {
                key: float(value) for key, value in re.findall(
                    r"([A-Za-z0-9_/]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                    line)
            }
    expected_steps = list(range(1, 129))
    assert sorted(console) == expected_steps

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    directory = run / "tensorboard"
    event_files = sorted(directory.glob("events.out.tfevents.*"))
    assert event_files, "Missing TensorBoard event file"
    reader = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    reader.Reload()
    available = reader.Tags()["scalars"]
    rows = {step: {"step": step} for step in expected_steps}
    maximum_console_difference = 0.0
    compared_values = 0
    for tag in TAGS:
        assert tag in available, "Missing scalar: " + tag
        events = reader.Scalars(tag)
        assert [event.step for event in events] == expected_steps, tag
        for event in events:
            value = float(event.value)
            assert math.isfinite(value), (tag, event.step)
            rows[event.step][tag] = value
            if tag in console[event.step]:
                difference = abs(value - console[event.step][tag])
                maximum_console_difference = max(maximum_console_difference, difference)
                # Console rounds a Python float to three decimal places; the
                # event stores float32. Account for both representations.
                tolerance = 0.0005 + max(abs(value), 1.0) * 2 ** -24 + 1e-12
                assert difference <= tolerance, (tag, event.step, difference)
                compared_values += 1
    assert any(row["actor/grad_norm"] != 0 for row in rows.values())
    if args.arm == "candidate":
        assert all(row["critic/advantages/min"] >= 0 for row in rows.values())
    assert all(row["agent_reward/semantic_reward_mean"] == 0 for row in rows.values())
    assert all(row["agent_reward/success_floor_mean"] == 0 for row in rows.values())

    report = {
        "schema": "positive_trajectory_tensorboard_training_export_v1",
        "run": name, "arm": args.arm, "seed": args.seed,
        "expected_steps": 128, "group_size": 4,
        "completed_at_utc": (run / "completed").read_text().strip(),
        "config": config_text,
        "config_sha256": sha256(config_path),
        "exporter_sha256": sha256(Path(__file__)),
        "train_log_sha256": sha256(log_path),
        "training_audit_sha256": sha256(audit_path),
        "event_files": [{"name": p.name, "bytes": p.stat().st_size,
                         "sha256": sha256(p)} for p in event_files],
        "scalar_tags": list(TAGS),
        "console_compared_values": compared_values,
        "max_absolute_difference_from_three_decimal_console": maximum_console_difference,
        "console_comparison_tolerance": "0.0005 + max(abs(event_value), 1) * 2**-24 + 1e-12",
        "nonzero_actor_gradient_steps": sum(row["actor/grad_norm"] != 0 for row in rows.values()),
        "nonzero_kl_loss_steps": sum(row["actor/kl_loss"] != 0 for row in rows.values()),
        "interpretation": (
            "Stored TensorBoard scalar precision, usually float32, exceeds the three-decimal "
            "console output. These are on-policy training diagnostics, not held-out navigation "
            "metrics or semantic-verifier accuracy. The goal-reached reason tag is preserved "
            "verbatim; it is not an independently evaluated SR. This export changes no gate, "
            "training configuration, checkpoint, or inference protocol."
        ),
        "steps": [rows[step] for step in expected_steps],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "run", "expected_steps", "console_compared_values",
        "max_absolute_difference_from_three_decimal_console",
        "nonzero_actor_gradient_steps", "nonzero_kl_loss_steps")}))


if __name__ == "__main__":
    main()
