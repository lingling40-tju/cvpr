"""Small behavioral checks for the staged confidence-pair reward."""

import json
from pathlib import Path
import unittest

from analyze_confidence_gate import gated_votes, load_part
from confident_pair_reward import group_relative_adjustments


def item(episode, raw, mode="stopped but goal not reached.", success=False):
    return {"episode_id": episode, "end_reason": mode,
            "task_success": success,
            "reward_components": {"fused_bonus": 0.0},
            "fused_reward": {"status": "disabled"} if success else
                            {"status": "ok", "raw": raw, "bonus": 0.0}}


class ConfidentPairRewardTest(unittest.TestCase):
    def test_high_margin_votes_and_zero_sum(self):
        infos = [item("one", raw) for raw in (0, 1, 6, 9)]
        removed, ordinal, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(removed, [0.0] * 4)
        for actual, expected in zip(ordinal, (-1 / 3, -1 / 6, 1 / 6, 1 / 3)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(ordinal), 0)
        self.assertEqual(counts["confident_pairs"], 3)
        self.assertEqual(counts["ordinal_active_groups"], 1)

    def test_low_margin_pairs_are_silent(self):
        infos = [item("one", raw) for raw in (0, 1, 2, 3)]
        _, ordinal, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(ordinal, [0.0] * 4)
        self.assertEqual(counts["confident_pairs"], 0)

    def test_modes_do_not_compete(self):
        stop = "stopped but goal not reached."
        timeout = "number of turns exceeded."
        infos = [item("one", 0, stop), item("one", 10, stop),
                 item("one", 20, timeout), item("one", 100, timeout)]
        _, ordinal, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(ordinal, [-0.5, 0.5, -0.5, 0.5])
        self.assertEqual(counts["same_mode_pairs"], 2)

    def test_mixed_success_keeps_outcome_reward(self):
        infos = [item("one", raw) for raw in (0, 10, 20)] + [
            item("one", 0, success=True)]
        _, ordinal, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(ordinal, [0.0] * 4)
        self.assertEqual(counts["all_failure_groups"], 0)

    def test_missing_eligible_teacher_score_fails(self):
        infos = [item("one", raw) for raw in (0, 10, 20, 30)]
        infos[0]["fused_reward"] = {"status": "error"}
        with self.assertRaises(ValueError):
            group_relative_adjustments(infos, 4)

    def test_rejects_nonstandard_group_size(self):
        with self.assertRaises(ValueError):
            group_relative_adjustments([item("one", 0)] * 4, 8)

    def test_exact_parity_with_frozen_train_scene_cache(self):
        base = Path(__file__).resolve().parents[1] / "ordinal_progress/policy_preference"
        root = base / "qwen3_route_match"
        groups = json.loads((base / "group_relative_manifest.json").read_text())
        policy = json.loads((root / "policy_manifest.json").read_text())
        for part, expected_groups in (("fit", 33), ("development", 7)):
            all_failure, _, _ = load_part(root, groups, policy, part)
            self.assertEqual(len(all_failure), expected_groups)
            for (_, eid), rows in all_failure.items():
                infos = [item(eid, row["score"], row["mode"]) for row in rows]
                _, online, _ = group_relative_adjustments(infos, 4)
                offline = gated_votes(rows, 5.5)
                for row, value in zip(rows, online):
                    self.assertAlmostEqual(value, offline[row["record_id"]])


if __name__ == "__main__":
    unittest.main()
