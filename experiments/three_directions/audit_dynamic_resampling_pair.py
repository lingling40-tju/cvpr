"""Audit same-row success-triggered resampling against destination control."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def main() -> None:
    steps, seed = map(int, sys.argv[1:3])
    assert steps in (2, 64) and seed == 11
    root = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
    name = f'three_directions_dynamic_resampling_{steps}step_seed{seed}'
    candidate_root = root / 'verl_checkpoints' / name
    control_root = root / 'verl_checkpoints/three_directions_branch_control_64step'
    run_dir = root / 'runlogs' / name
    dataset = root / 'data/branch_pilot_train.parquet'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == (
        'a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3')
    assert (candidate_root / f'global_step_{steps}/actor/huggingface/config.json').exists()
    config = (run_dir / 'config.txt').read_text()
    assert 'mode=dynamic_resampling ' in config
    assert 'rollout_n=2 max_extra_attempts=2' in config
    candidate = [json.loads(line) for line in (candidate_root / 'rollout.jsonl').read_text().splitlines()]
    control = [json.loads(line) for line in (control_root / 'rollout.jsonl').read_text().splitlines()[:steps]]
    assert [r['step'] for r in candidate] == [r['step'] for r in control] == list(range(1, steps + 1))
    seen = set()
    diverse_groups = varied_groups = dynamic_final_rollouts = 0
    all_tied_steps = []
    for c, b in zip(candidate, control):
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
        assert set(control_ids.values()) == {2}
        assert not seen.intersection(by_episode)
        seen.update(by_episode)
        step_varied = 0
        for items in by_episode.values():
            assert len(items) == 2
            signatures = [tuple(turn['response'] for turn in x['gen_traj']) for x in items]
            diverse_groups += signatures[0] != signatures[1]
            step_varied += items[0]['total_reward'] != items[1]['total_reward']
        varied_groups += step_varied
        if step_varied == 0:
            all_tied_steps.append(c['step'])
    assert len(seen) == 4 * steps and diverse_groups > 0 and varied_groups > 0
    log = (run_dir / 'train.log').read_text(errors='replace')
    attempts = [(int(i), int(count)) for i, count in re.findall(
        r'\[Trajectory Rollout \(Dynamic Sampling #(\d+)\)\] turn=0, bz=(\d+)', log)]
    assert attempts and all(1 <= i <= 2 and 1 <= count <= 4 for i, count in attempts)
    extra_rollouts = 2 * sum(count for _, count in attempts)
    assert dynamic_final_rollouts > 0
    exact_cost = steps == 64
    if exact_cost:
        assert 'ray_dedup_logs=0' in config
        assert '[repeated ' not in log
        assert dynamic_final_rollouts <= extra_rollouts
    grads = [float(x) for x in re.findall(r'actor/grad_norm:([0-9.eE+-]+)', log)]
    assert len(grads) >= steps and all(math.isfinite(x) and x >= 0 for x in grads[:steps])
    assert any(x > 1e-6 for x in grads[:steps])
    zero_grad_steps = [i + 1 for i, x in enumerate(grads[:steps]) if x == 0]
    assert set(zero_grad_steps).issubset(all_tied_steps)
    print(json.dumps({
        'seed': seed, 'steps': steps, 'rollout_n': 2,
        'matched_train_episode_sets_at_each_step': steps,
        'unique_train_episodes': len(seen),
        'candidate_final_rollouts': steps * 8,
        'control_rollouts': steps * 8,
        'logged_dynamic_attempts': len(attempts),
        'logged_additional_simulator_rollouts': extra_rollouts,
        'rollout_cost_exact_from_logs': exact_cost,
        'total_candidate_simulator_rollouts': steps * 8 + extra_rollouts if exact_cost else None,
        'final_rollouts_from_dynamic_sampling': dynamic_final_rollouts,
        'diverse_candidate_groups': diverse_groups,
        'nonzero_return_variance_candidate_groups': varied_groups,
        'all_tied_steps': all_tied_steps,
        'zero_grad_steps': zero_grad_steps,
        'actor_grad_norms': grads[:steps],
        'interpretation': 'Training-row and rollout-cost audit, not held-out navigation evidence.',
    }, indent=2))


if __name__ == '__main__':
    main()
