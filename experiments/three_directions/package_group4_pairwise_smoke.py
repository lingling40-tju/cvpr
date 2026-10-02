"""Export compact evidence from the completed 2-step GRPO pairing check."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    name = 'three_directions_group4_pairwise_2step_seed11'
    run = args.root / 'runlogs' / name
    source = args.root / 'verl_checkpoints' / name / 'rollout.jsonl'
    assert (run / 'completed').exists() and not (run / 'failed').exists()
    audit = json.loads((run / 'paired_train_audit.json').read_text())
    assert audit['mode'] == 'group4_pairwise' and audit['steps'] == 2
    rows = []
    records = [json.loads(line) for line in source.read_text().splitlines()]
    assert [x['step'] for x in records] == [1, 2]
    for record in records:
        for item in record['info']:
            rows.append({
                'step': record['step'],
                'episode_id': str(item['episode_id']),
                'grpo_uid': str(item['grpo_uid']),
                'total_reward': float(item['total_reward']),
                'task_success': bool(item['task_success']),
            })
    assert len(rows) == 32
    varied = 0
    for step in (1, 2):
        subset = [x for x in rows if x['step'] == step]
        by_episode = collections.Counter(x['episode_id'] for x in subset)
        by_uid = collections.defaultdict(list)
        for item in subset:
            by_uid[item['grpo_uid']].append(item)
        assert len(by_episode) == 4 and set(by_episode.values()) == {4}
        assert len(by_uid) == 8 and set(map(len, by_uid.values())) == {2}
        assert all(len({x['episode_id'] for x in pair}) == 1 for pair in by_uid.values())
        varied += sum(pair[0]['total_reward'] != pair[1]['total_reward']
                      for pair in by_uid.values())
    assert varied == audit['nonzero_return_variance_pairs']
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    compact = output / 'rollout_groups.jsonl'
    compact.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    for src, dst in ((run / 'paired_train_audit.json', output / 'paired_train_audit.json'),
                     (run / 'config.txt', output / 'config.txt')):
        dst.write_bytes(src.read_bytes())
    package = {
        'mode': 'group4_pairwise', 'seed': 11, 'steps': 2,
        'source_rollout_sha256': sha(source),
        'rollout_groups_sha256': sha(compact),
        'train_audit_sha256': sha(output / 'paired_train_audit.json'),
        'config_sha256': sha(output / 'config.txt'),
        'rollouts': len(rows), 'nonzero_return_variance_pairs': varied,
    }
    (output / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
