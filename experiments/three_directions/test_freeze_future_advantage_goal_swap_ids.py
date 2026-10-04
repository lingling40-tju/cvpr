"""Check deterministic farthest-goal ID selection without route outcomes."""

from freeze_future_advantage_goal_swap_ids import select_pairs


def episode(eid, scene, start, goal):
    return {"episode_id": eid, "scene_id": scene,
            "start_position": start, "start_rotation": [0, 0, 0, 1],
            "goals": [{"position": goal}]}


rows = [
    episode(3, "audit", [0, 0, 0], [0, 0, 0]),
    episode(1, "audit", [0, 0, 0], [0, 0, 2]),
    episode(2, "audit", [0, 0, 0], [0, 0, 5]),
    episode(4, "audit", [3, 0, 0], [0, 0, 0]),
    episode(5, "fit", [0, 0, 0], [0, 0, 9]),
]
selected = select_pairs(rows, {"audit"})
assert len(selected) == 1
assert selected[0]["episode_ids"] == [2, 3]
assert selected[0]["goal_separation_m"] == 5.0
assert selected[0]["pair_id"] == "goal_swap_001"
print("goal-swap ID source selection passed")
