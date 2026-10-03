"""Behavioral checks for the premature-STOP group-four diagnostic."""

from __future__ import annotations

import unittest
from collections import Counter

from audit_stop_pair_train import audit_group
from stop_pair_group4_reward import group_relative_adjustments


def item(distance: float, reason: str, *, success: bool = False) -> dict:
    return {"episode_id": "one", "distance_to_goal": distance,
            "end_reason": reason, "task_success": success,
            "reward_components": {"fused_bonus": 0.0}}


STOP = "stopped but goal not reached."
CAP = "number of turns exceeded."


class StopPairGroupFourTest(unittest.TestCase):
    def test_only_clear_continuation_pairs_receive_credit(self):
        infos = [item(9.0, STOP), item(6.0, CAP),
                 item(8.5, STOP), item(9.2, CAP)]
        removed, values, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(removed, [0.0] * 4)
        self.assertEqual(values, [-1 / 6, 2 / 6, -1 / 6, 0.0])
        self.assertEqual(counts["continuation_over_stop_pairs"], 2)
        self.assertEqual(counts["active_groups"], 1)
        self.assertAlmostEqual(sum(values), 0)

    def test_success_group_preserves_outcome(self):
        infos = [item(2.0, "successfully reached the goal.", success=True),
                 item(9.0, STOP), item(6.0, CAP), item(8.0, CAP)]
        _, values, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(values, [0.0] * 4)
        self.assertEqual(counts["all_failure_groups"], 0)

    def test_ambiguous_near_goal_continuation_is_excluded(self):
        infos = [item(9.0, STOP), item(2.5, CAP),
                 item(8.0, STOP), item(10.0, CAP)]
        _, values, counts = group_relative_adjustments(infos, 4)
        self.assertEqual(values, [0.0] * 4)
        self.assertEqual(counts["active_groups"], 0)

    def test_fail_closed_on_group_and_bonus(self):
        infos = [item(9.0, STOP), item(6.0, CAP),
                 item(8.5, STOP), item(9.2, CAP)]
        with self.assertRaisesRegex(ValueError, "group size four"):
            group_relative_adjustments(infos, 8)
        infos[0]["reward_components"]["fused_bonus"] = 0.1
        with self.assertRaisesRegex(ValueError, "teacher bonus"):
            group_relative_adjustments(infos, 4)

    def test_independent_audit_catches_reward_corruption(self):
        infos = [item(9.0, STOP), item(6.0, CAP),
                 item(8.5, STOP), item(9.2, CAP)]
        _, values, _ = group_relative_adjustments(infos, 4)
        for info, vote in zip(infos, values):
            info["fused_reward"] = {"status": "disabled"}
            info["stop_pair_diagnostic"] = {"removed_bonus": 0.0,
                                            "applied_ordinal": vote}
            info["reward_components"].update({
                "stop_pair_ordinal": vote, "success_reward": 0.0,
                "success_floor": 0.0, "ndtw_reward": 0.0,
                "semantic_reward": 0.0,
            })
            info["total_reward"] = vote
        audit_group(infos, Counter())
        infos[0]["reward_components"]["stop_pair_ordinal"] += 0.01
        with self.assertRaisesRegex(ValueError, "reward wiring mismatch"):
            audit_group(infos, Counter())


if __name__ == "__main__":
    unittest.main()
