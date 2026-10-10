#!/usr/bin/env python3
"""CPU-only, same-prompt leave-one-trajectory-out HAPO kernel audit."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def returns(rewards, gamma):
    out = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        out[t] = acc
    return out


def run(root: Path, seeds=(11, 22, 33), gamma=0.95, sigmas=(0.0, 1.0, 2.0, 4.0, math.inf)):
    result = {
        "schema": "hapo_group4_kernel_prediction_preflight_v1",
        "scope": "CPU-only diagnostic on completed n4 train rollouts; no navigation evaluation",
        "gamma": gamma,
        "grouping": "same configured training episode ID within optimizer update; exactly four rollouts",
        "baseline": "leave-one-trajectory-out Gaussian kernel over absolute turn index",
        "updates_required_per_seed": 128,
        "seeds": {},
    }
    for seed in seeds:
        path = root / f"oracle_turnwise_exact512_128_seed{seed}" / "rollout.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        steps = 0
        groups = 0
        trajectory_count = 0
        active_turn_count = 0
        mse_sq = {str(s): [] for s in sigmas}
        for line_no, line in enumerate(path.open(), 1):
            obj = json.loads(line)
            steps += 1
            grouped = {}
            for info in obj["info"]:
                grouped.setdefault(str(info["episode_id"]), []).append(info)
            if len(grouped) != len(obj["info"]) // 4 or any(len(x) != 4 for x in grouped.values()):
                raise ValueError(f"seed {seed} update {obj.get('step')} not exact n=4 groups")
            for episode_id, infos in grouped.items():
                groups += 1
                trajectory_count += len(infos)
                group_returns = []
                for info in infos:
                    traj = info["gen_traj"]
                    if not traj or any("oracle_turn_progress" not in turn for turn in traj):
                        raise ValueError(f"missing oracle turn signal, seed {seed} line {line_no}")
                    group_returns.append(returns([turn["oracle_turn_progress"] for turn in traj], gamma))
                for i, own in enumerate(group_returns):
                    for t, observed_return in enumerate(own):
                        active_turn_count += 1
                        for sigma in sigmas:
                            weighted_sum = 0.0
                            weight_sum = 0.0
                            for j, other in enumerate(group_returns):
                                if j == i:
                                    continue
                                for other_t, value in enumerate(other):
                                    if sigma == 0.0:
                                        weight = 1.0 if other_t == t else 0.0
                                    elif math.isinf(sigma):
                                        weight = 1.0
                                    else:
                                        weight = math.exp(-((t - other_t) ** 2) / (2.0 * sigma * sigma))
                                    weighted_sum += weight * value
                                    weight_sum += weight
                            if weight_sum == 0.0:
                                continue
                            prediction = weighted_sum / weight_sum
                            mse_sq[str(sigma)].append((observed_return - prediction) ** 2)
        if steps != 128:
            raise ValueError(f"seed {seed} has {steps} updates, expected 128")
        metrics = {}
        for sigma in sigmas:
            values = mse_sq[str(sigma)]
            metrics[str(sigma)] = {
                "count": len(values),
                "rmse": math.sqrt(statistics.fmean(values)),
            }
        local = metrics["2.0"]["rmse"]
        uniform = metrics["inf"]["rmse"]
        result["seeds"][str(seed)] = {
            "source": str(path),
            "source_sha256": sha256(path),
            "updates": steps,
            "n4_groups": groups,
            "trajectories": trajectory_count,
            "active_turns": active_turn_count,
            "holdout_return_rmse_by_sigma": metrics,
            "sigma2_rmse_change_vs_uniform_percent": 100.0 * (local / uniform - 1.0),
        }
    changes = [result["seeds"][str(s)]["sigma2_rmse_change_vs_uniform_percent"] for s in seeds]
    result["summary"] = {
        "sigma2_rmse_lower_than_uniform_all_seeds": all(x < 0 for x in changes),
        "mean_sigma2_rmse_change_vs_uniform_percent": statistics.fmean(changes),
        "interpretation": "Predictive baseline diagnostic only; does not establish a policy or navigation gain.",
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = run(args.rollout_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
