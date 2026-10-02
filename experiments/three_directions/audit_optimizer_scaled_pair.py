"""Audit 128-step group-size or KL runs against matched destination controls."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


DATASET_SHA = '2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea'


def read_steps(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1, 129))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('group4', 'kl_anchor'), required=True)
    parser.add_argument('--seed', type=int, choices=(11, 22, 33), required=True)
    args = parser.parse_args()
    root = args.root
    suffix = '' if args.seed == 11 else f'_seed{args.seed}'
    name = f'three_directions_{args.mode}_128step{suffix}'
    control_name = f'three_directions_branch_control_128step{suffix}'
    candidate_root = root / 'verl_checkpoints' / name
    control_root = root / 'verl_checkpoints' / control_name
    run_dir = root / 'runlogs' / name
    control_run = root / 'runlogs' / control_name
    assert (control_run / 'completed').exists()
    assert (candidate_root / 'global_step_128/actor/huggingface/config.json').exists()
    dataset_path = root / 'data/branch_scale512_train.parquet'
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert digest == DATASET_SHA
    dataset = pd.read_parquet(dataset_path)
    dataset_info = list(dataset['extra_info'])
    dataset_ids = [str(item['episode_id']) for item in dataset_info]
    assert len(dataset_ids) == len(set(dataset_ids)) == 512
    assert all(item['split'] == 'train' for item in dataset_info)
    config = (run_dir / 'config.txt').read_text()
    assert f'mode={args.mode} steps=128 ' in config
    assert f'seed={args.seed} ' in config
    assert f'dataset_sha256={DATASET_SHA} ' in config
    expected_n = 4 if args.mode == 'group4' else 2
    assert f'rollout_n={expected_n}' in config
    if args.mode == 'kl_anchor':
        assert 'actor_kl_coef=0.001' in config
    candidate = read_steps(candidate_root / 'rollout.jsonl')
    control = read_steps(control_root / 'rollout.jsonl')
    seen = set()
    varied_groups = diverse_groups = 0
    all_tied_steps = []
    for step_candidate, step_control in zip(candidate, control):
        by_episode = collections.defaultdict(list)
        for item in step_candidate['info']:
            by_episode[str(item['episode_id'])].append(item)
            assert item['env_global_step'] == item['env_local_step']
            assert math.isfinite(float(item['total_reward']))
            parts = item['reward_components']
            assert float(parts['ndtw_reward']) == 0
            assert float(parts['semantic_reward']) == 0
            assert 'geodesic_progress_reward' not in parts
        control_ids = collections.Counter(str(item['episode_id']) for item in step_control['info'])
        assert len(by_episode) == len(control_ids) == 4
        assert set(by_episode) == set(control_ids)
        expected_ids = set(dataset_ids[4 * (step_candidate['step'] - 1):4 * step_candidate['step']])
        assert set(by_episode) == expected_ids
        assert set(control_ids.values()) == {2}
        assert not seen.intersection(by_episode)
        seen.update(by_episode)
        step_varied = 0
        for items in by_episode.values():
            assert len(items) == expected_n
            signatures = [tuple(turn['response'] for turn in x['gen_traj']) for x in items]
            diverse_groups += len(set(signatures)) > 1
            step_varied += len({float(x['total_reward']) for x in items}) > 1
        varied_groups += step_varied
        if step_varied == 0:
            all_tied_steps.append(step_candidate['step'])
    assert seen == set(dataset_ids) and diverse_groups > 0 and varied_groups > 0
    log = (run_dir / 'train.log').read_text()
    grads = [float(x) for x in re.findall(r'actor/grad_norm:([0-9.eE+-]+)', log)]
    assert len(grads) >= 128 and all(math.isfinite(x) and x >= 0 for x in grads[:128])
    assert any(x > 1e-6 for x in grads[:128])
    zero_grad_steps = [i + 1 for i, x in enumerate(grads[:128]) if x == 0]
    assert set(zero_grad_steps).issubset(all_tied_steps)
    result = {
        'mode': args.mode, 'seed': args.seed, 'steps': 128,
        'dataset_sha256': DATASET_SHA,
        'matched_train_episode_sets_at_each_step': 128,
        'unique_train_episodes': len(seen),
        'candidate_rollouts': 128 * 4 * expected_n,
        'control_rollouts': 128 * 8,
        'diverse_candidate_groups': diverse_groups,
        'nonzero_return_variance_candidate_groups': varied_groups,
        'all_tied_steps': all_tied_steps,
        'zero_grad_steps': zero_grad_steps,
        'actor_grad_norms': grads[:128],
        'interpretation': 'Train-row and optimizer audit, not held-out navigation evidence.',
    }
    if args.mode == 'kl_anchor':
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        kl = [float(x) for x in re.findall(r'actor/kl_loss:([0-9.eE+-]+)', log)]
        assert len(kl) >= 128 and all(math.isfinite(x) for x in kl[:128])
        event_files = list((run_dir / 'tensorboard').glob('events.out.tfevents.*'))
        assert event_files
        kl_by_step, coef_by_step = {}, {}
        for path in sorted(event_files):
            events = EventAccumulator(str(path), size_guidance={'scalars': 0})
            events.Reload()
            assert {'actor/kl_loss', 'actor/kl_coef'} <= set(events.Tags()['scalars'])
            kl_by_step.update((x.step, x.value) for x in events.Scalars('actor/kl_loss'))
            coef_by_step.update((x.step, x.value) for x in events.Scalars('actor/kl_coef'))
        tb_kl = [kl_by_step[i] for i in range(1, 129)]
        tb_coef = [coef_by_step[i] for i in range(1, 129)]
        assert all(math.isfinite(x) for x in tb_kl)
        assert all(math.isclose(x, 0.001, abs_tol=1e-7) for x in tb_coef)
        result['actor_kl_coef'] = 0.001
        result['actor_kl_losses_console_rounded'] = kl[:128]
        result['actor_kl_losses_tensorboard'] = tb_kl
        result['actor_kl_nonzero_tensorboard_steps'] = sum(abs(x) > 1e-9 for x in tb_kl)
        result['actor_kl_coef_tensorboard'] = tb_coef
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
