"""Contract checks for mode-stratified group-four reward transformation."""

import unittest

from mode_stratified_reward import mode_stratified_adjustments


def row(episode, mode, score=None, success=False):
    bonus = 0.0 if success else 0.5
    return {
        "episode_id": episode,
        "task_success": success,
        "end_reason": mode,
        "reward_components": {"fused_bonus": bonus},
        "fused_reward": ({"status": "disabled"} if success else
                         {"status": "ok", "raw": score, "bonus": bonus}),
    }


class ModeRewardTest(unittest.TestCase):
    def test_same_mode_ranks_without_cross_mode_advantage(self):
        infos = [
            row(1, "stopped but goal not reached.", .1),
            row(1, "number of turns exceeded.", .8),
            row(1, "stopped but goal not reached.", .4),
            row(1, "number of turns exceeded.", .2),
        ]
        removed, ranks, report = mode_stratified_adjustments(infos, 4)
        self.assertEqual(removed, [.5] * 4)
        self.assertEqual(ranks, [-.5, .5, .5, -.5])
        self.assertEqual(report["ordinal_active_groups"], 1)
        self.assertEqual(report["same_mode_pairs"], 2)

    def test_success_group_keeps_only_outcome_reward(self):
        infos = [row(2, "successfully reached the goal.", success=True)]
        infos += [row(2, "number of turns exceeded.", score)
                  for score in (.1, .2, .3)]
        removed, ranks, report = mode_stratified_adjustments(infos, 4)
        self.assertEqual(removed, [0, .5, .5, .5])
        self.assertEqual(ranks, [0] * 4)
        self.assertEqual(report["all_failure_groups"], 0)

    def test_ties_and_singletons(self):
        infos = [
            row(3, "stopped but goal not reached.", .2),
            row(3, "stopped but goal not reached.", .2),
            row(3, "number of turns exceeded.", .4),
            row(3, "unexpected format.", .8),
        ]
        _, ranks, report = mode_stratified_adjustments(infos, 4)
        self.assertEqual(ranks, [0] * 4)
        self.assertEqual(report["ordinal_active_groups"], 0)

    def test_rejects_unmatched_groups(self):
        with self.assertRaises(ValueError):
            mode_stratified_adjustments([row(4, "number of turns exceeded.", .1)], 4)


if __name__ == "__main__":
    unittest.main()
