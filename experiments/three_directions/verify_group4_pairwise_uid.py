"""Preflight the prospective compute-matched four-rollout GRPO grouping."""

from group4_pairwise_uid import split_group4_uids


def main() -> None:
    sources = [f"episode-{i}" for i in range(4)]
    repeated = [uid for uid in sources for _ in range(4)]
    grouped = split_group4_uids(repeated)
    assert len(grouped) == len(repeated) == 16
    assert len(set(grouped)) == 8
    for i in range(4):
        assert grouped[4 * i] == grouped[4 * i + 1]
        assert grouped[4 * i + 2] == grouped[4 * i + 3]
        assert grouped[4 * i] != grouped[4 * i + 2]
    assert len({grouped[i] for i in (0, 4, 8, 12)}) == 4
    for bad in ([], ["a"], ["a", "a", "a", "b"], ["a"] * 8):
        try:
            split_group4_uids(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed episode-major batch: {bad}")
    print("four rollouts per episode become two distinct two-rollout GRPO groups")


if __name__ == "__main__":
    main()
