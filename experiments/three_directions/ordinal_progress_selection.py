"""Shared deterministic pilot selection for ordinal progress data passes."""

from __future__ import annotations


def select_episode_ids(manifest: dict, subset: str, limit: int = 0) -> list[int]:
    rows = manifest["subsets"][subset]
    all_ids = rows["episode_ids"]
    if limit == 0:
        return all_ids
    if not 0 < limit <= len(all_ids):
        raise ValueError(f"invalid {subset} limit {limit}")
    pair_count = min(len(rows["pairs"]), limit // 4)
    all_paired = {episode_id for pair in rows["pairs"]
                  for episode_id in (pair["left"], pair["right"])}
    selected = {episode_id for pair in rows["pairs"][:pair_count]
                for episode_id in (pair["left"], pair["right"])}
    filler = [episode_id for episode_id in all_ids if episode_id not in all_paired]
    selected.update(filler[:limit - len(selected)])
    if len(selected) != limit or not selected.issubset(set(all_ids)):
        raise ValueError(f"cannot select {limit} episodes from {subset}")
    return sorted(selected)
