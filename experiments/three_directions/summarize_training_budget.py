"""Summarize the three completed, matched 128-step training audits."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "scale_budget"
SEEDS = (11, 22, 33)
ARMS = ("branch", "branch_control")
FIELDS = (
    "generated_grouped_action_commands",
    "replayed_grouped_action_commands",
    "total_grouped_action_commands",
    "wrapper_elapsed_hours",
    "nonzero_return_variance_groups",
    "nonzero_actor_gradient_steps",
)
DATASET_SHA256 = "2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea"


def load_seed(seed):
    audit = json.loads((DATA / f"seed{seed}_pair_audit.json").read_text())
    assert audit["seed"] == seed and audit["audited_steps"] == 128
    assert audit["partial"] is False and audit["dataset_rows"] == 512
    assert audit["dataset_sha256"] == DATASET_SHA256
    assert audit["matched_episode_sets_at_each_step"] == 128
    result = {}
    for arm in ARMS:
        arm_audit = audit["arms"][arm]
        validation = json.loads((DATA / f"seed{seed}_{arm}_validation.json").read_text())
        assert arm_audit["episode_count"] == 512
        assert arm_audit["rollouts_per_episode"] == 2
        assert arm_audit["training_rollouts"] == 1024
        assert validation["steps"] == 128
        assert validation["groups"] == validation["diverse_trajectory_groups"] == 512
        assert len(validation["actor_grad_norms"]) == 128
        assert (
            arm_audit["generated_grouped_action_commands"]
            + arm_audit["replayed_grouped_action_commands"]
            == arm_audit["total_grouped_action_commands"]
        )
        result[arm] = {
            field: arm_audit[field]
            for field in FIELDS
            if field in arm_audit
        }
        result[arm]["nonzero_return_variance_groups"] = validation[
            "nonzero_return_variance_groups"
        ]
        result[arm]["nonzero_actor_gradient_steps"] = sum(
            float(value) > 0 for value in validation["actor_grad_norms"]
        )
    assert result["branch"]["replayed_grouped_action_commands"] == 6088
    assert result["branch_control"]["replayed_grouped_action_commands"] == 0
    return result


def main():
    per_seed = {str(seed): load_seed(seed) for seed in SEEDS}
    aggregate = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for field in FIELDS:
            values = [per_seed[str(seed)][arm][field] for seed in SEEDS]
            aggregate[arm][field] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
            }
    output = {
        "scope": "128 optimizer steps, 512 matched train episodes and 1024 rollouts per arm and seed",
        "unit": "grouped action commands; forward commands may execute multiple simulator steps",
        "elapsed_definition": "wrapper time from config write through training, checkpoint save, and validation; not GPU-hours",
        "dataset_sha256": DATASET_SHA256,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "interpretation": "Training budget and signal diagnostics only; no held-out navigation outcomes.",
    }
    path = DATA / "three_seed_training_budget.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
