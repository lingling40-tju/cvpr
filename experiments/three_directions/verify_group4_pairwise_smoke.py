"""Independently check the compact two-step GRPO pairing package."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(sys.argv[1])
    package = json.loads((root / 'package.json').read_text())
    audit = json.loads((root / 'paired_train_audit.json').read_text())
    rows = [json.loads(line) for line in (root / 'rollout_groups.jsonl').read_text().splitlines()]
    assert package['mode'] == audit['mode'] == 'group4_pairwise'
    assert package['seed'] == audit['seed'] == 11
    assert package['steps'] == audit['steps'] == 2
    for key, file in (('rollout_groups_sha256', 'rollout_groups.jsonl'),
                      ('train_audit_sha256', 'paired_train_audit.json'),
                      ('config_sha256', 'config.txt')):
        assert package[key] == sha(root / file)
    assert package['rollouts'] == audit['candidate_rollouts'] == len(rows) == 32
    assert 'rollout_n=4 pairwise_uid=1' in (root / 'config.txt').read_text()
    all_episode_ids = set()
    varied_pairs = 0
    for step in (1, 2):
        subset = [x for x in rows if x['step'] == step]
        by_episode = collections.defaultdict(list)
        by_uid = collections.defaultdict(list)
        for item in subset:
            by_episode[item['episode_id']].append(item)
            by_uid[item['grpo_uid']].append(item)
        assert len(by_episode) == 4 and set(map(len, by_episode.values())) == {4}
        assert len(by_uid) == 8 and set(map(len, by_uid.values())) == {2}
        for uid, pair in by_uid.items():
            assert uid.endswith((':pair0', ':pair1'))
            assert len({x['episode_id'] for x in pair}) == 1
            varied_pairs += pair[0]['total_reward'] != pair[1]['total_reward']
        for items in by_episode.values():
            uids = {x['grpo_uid'] for x in items}
            assert len(uids) == 2
            assert {uid.rsplit(':pair', 1)[1] for uid in uids} == {'0', '1'}
            assert len({uid.rsplit(':pair', 1)[0] for uid in uids}) == 1
        assert not all_episode_ids.intersection(by_episode)
        all_episode_ids.update(by_episode)
    assert len(all_episode_ids) == audit['unique_train_episodes'] == 8
    assert varied_pairs == audit['nonzero_return_variance_pairs'] == package['nonzero_return_variance_pairs']
    assert audit['matched_train_episode_sets_at_each_step'] == 2
    assert len(audit['actor_grad_norms']) == 2 and all(x > 0 for x in audit['actor_grad_norms'])
    print(f'verified 2 steps, 8 episodes, 16 paired groups, {varied_pairs} varied pairs')


if __name__ == '__main__':
    main()
