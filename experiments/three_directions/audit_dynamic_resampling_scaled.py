"""Audit three-seed 128-step resampling against exact destination controls."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


DATASET_SHA = '2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea'


def main() -> None:
    seed = int(sys.argv[1])
    assert seed in (11, 22, 33)
    root = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
    suffix = '' if seed == 11 else f'_seed{seed}'
    name = f'three_directions_dynamic_resampling_128step{suffix}'
    control = f'three_directions_branch_control_128step{suffix}'
    candidate_root = root / 'verl_checkpoints' / name
    control_root = root / 'verl_checkpoints' / control
    run_dir = root / 'runlogs' / name
    assert (root / 'runlogs' / control / 'completed').exists()
    assert (candidate_root / 'global_step_128/actor/huggingface/config.json').exists()
    dataset_path = root / 'data/branch_scale512_train.parquet'
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == DATASET_SHA
    dataset_info = list(pd.read_parquet(dataset_path)['extra_info'])
    dataset_ids = [str(item['episode_id']) for item in dataset_info]
    assert len(dataset_ids) == len(set(dataset_ids)) == 512
    assert all(item['split'] == 'train' for item in dataset_info)
    config = (run_dir / 'config.txt').read_text()
    assert f'mode=dynamic_resampling steps=128 gpus=0,1 seed={seed} ' in config
    assert f'dataset_sha256={DATASET_SHA} ' in config
    assert 'service=http://127.0.0.1:5015 rollout_n=2 max_extra_attempts=2 ray_dedup_logs=0' in config
    candidate = [json.loads(line) for line in (candidate_root / 'rollout.jsonl').read_text().splitlines()]
    baseline = [json.loads(line) for line in (control_root / 'rollout.jsonl').read_text().splitlines()]
    assert [r['step'] for r in candidate] == [r['step'] for r in baseline] == list(range(1, 129))
    seen = set()
    diverse_groups = varied_groups = dynamic_final_rollouts = 0
    all_tied_steps = []
    for c, b in zip(candidate, baseline):
        by_episode = collections.defaultdict(list)
        for item in c['info']:
            by_episode[str(item['episode_id'])].append(item)
            assert item['env_global_step'] == item['env_local_step']
            assert math.isfinite(float(item['total_reward']))
            parts = item['reward_components']
            assert float(parts['ndtw_reward']) == 0
            assert float(parts['semantic_reward']) == 0
            assert 'geodesic_progress_reward' not in parts
            dynamic_final_rollouts += bool(item.get('from_dynamic_sampling', False))
        control_ids = collections.Counter(str(item['episode_id']) for item in b['info'])
        assert len(by_episode) == len(control_ids) == 4
        assert set(by_episode) == set(control_ids)
        expected_ids = set(dataset_ids[4 * (c['step'] - 1):4 * c['step']])
        assert set(by_episode) == expected_ids
        assert set(control_ids.values()) == {2}
        assert not seen.intersection(by_episode)
        seen.update(by_episode)
        step_varied = 0
        for pair in by_episode.values():
            assert len(pair) == 2
            signatures = [tuple(turn['response'] for turn in x['gen_traj']) for x in pair]
            diverse_groups += signatures[0] != signatures[1]
            step_varied += pair[0]['total_reward'] != pair[1]['total_reward']
        varied_groups += step_varied
        if step_varied == 0:
            all_tied_steps.append(c['step'])
    assert seen == set(dataset_ids) and diverse_groups > 0 and varied_groups > 0
    log = (run_dir / 'train.log').read_text(errors='replace')
    assert '[repeated ' not in log
    attempts = [(int(i), int(count)) for i, count in re.findall(
        r'\[Trajectory Rollout \(Dynamic Sampling #(\d+)\)\] turn=0, bz=(\d+)', log)]
    assert attempts and all(1 <= i <= 2 and 1 <= count <= 4 for i, count in attempts)
    extra_rollouts = 2 * sum(count for _, count in attempts)
    assert 0 < dynamic_final_rollouts <= extra_rollouts
    grads = [float(x) for x in re.findall(r'actor/grad_norm:([0-9.eE+-]+)', log)]
    assert len(grads) >= 128 and all(math.isfinite(x) and x >= 0 for x in grads[:128])
    assert any(x > 1e-6 for x in grads[:128])
    zero_grad_steps = [i + 1 for i, x in enumerate(grads[:128]) if x == 0]
    assert set(zero_grad_steps).issubset(all_tied_steps)
    print(json.dumps({
        'mode': 'dynamic', 'seed': seed, 'steps': 128,
        'dataset_sha256': DATASET_SHA,
        'matched_train_episode_sets_at_each_step': 128,
        'unique_train_episodes': len(seen),
        'candidate_final_rollouts': 1024,
        'control_rollouts': 1024,
        'logged_dynamic_attempts': len(attempts),
        'additional_simulator_rollouts': extra_rollouts,
        'total_candidate_simulator_rollouts': 1024 + extra_rollouts,
        'final_rollouts_from_dynamic_sampling': dynamic_final_rollouts,
        'diverse_candidate_groups': diverse_groups,
        'nonzero_return_variance_candidate_groups': varied_groups,
        'all_tied_steps': all_tied_steps,
        'zero_grad_steps': zero_grad_steps,
        'actor_grad_norms': grads[:128],
        'interpretation': 'Same-row and simulator-cost audit, not held-out navigation evidence.',
    }, indent=2))


if __name__ == '__main__':
    main()
