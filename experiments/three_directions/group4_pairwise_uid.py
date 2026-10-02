"""Split four interleaved rollouts into two GRPO pairs per train episode.

This helper is for a prospective compute-matched ablation. It must be copied
into ``verl/trainer/ppo`` in an isolated source tree before using the patch.
"""

from __future__ import annotations

from collections.abc import Sequence


def split_group4_uids(repeated_uids: Sequence[str]) -> list[str]:
    """Return pair IDs for an episode-major, four-rollout repeated batch.

    The caller must use ``DataProto.repeat(repeat_times=4, interleave=True)``.
    UUIDs are assigned once per episode before that repeat. This function
    checks that each quartet has one unique source UUID and that distinct
    episodes do not accidentally share one.
    """
    if len(repeated_uids) == 0 or len(repeated_uids) % 4:
        raise ValueError("expected a nonempty batch of complete rollout quartets")
    result = []
    source_ids = set()
    for offset in range(0, len(repeated_uids), 4):
        quartet = [str(uid) for uid in repeated_uids[offset:offset + 4]]
        if len(set(quartet)) != 1:
            raise ValueError(f"rollout quartet at offset {offset} is not episode-major")
        source_id = quartet[0]
        if source_id in source_ids:
            raise ValueError(f"source UID is reused across episodes: {source_id}")
        source_ids.add(source_id)
        result.extend((f"{source_id}:pair0", f"{source_id}:pair0",
                       f"{source_id}:pair1", f"{source_id}:pair1"))
    if len(set(result)) != 2 * len(source_ids):
        raise AssertionError("pair UID collision")
    return result
