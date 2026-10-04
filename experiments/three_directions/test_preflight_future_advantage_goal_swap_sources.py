"""Check that alternative goals share the complete start pose."""

from preflight_future_advantage_goal_swap_sources import matching_ids


def episode(eid, start, rotation, goal):
    return {"episode_id": eid, "scene_id": "train_scene",
            "start_position": start, "start_rotation": rotation,
            "goals": [{"position": goal}]}


same_start = [0., 0., 0.]
same_rotation = [0., 0., 0., 1.]
examples = [
    episode(1, same_start, same_rotation, [0., 0., 2.]),
    episode(2, same_start, same_rotation, [0., 0., 5.]),
    episode(3, same_start, same_rotation, [0., 0., 2.]),
    episode(4, same_start, [0., 0., 1., 0.], [0., 0., 8.]),
    episode(5, [2., 0., 0.], same_rotation, [0., 0., 8.]),
]
result = matching_ids(examples)
assert result["start_groups"] == 3
assert result["start_groups_with_alternative_goal"] == 1
assert result["different_goal_pairs"] == 2
assert result["episode_ids_with_alternative_goal"] == 3
print("goal-swap start and distinct-goal grouping passed")
