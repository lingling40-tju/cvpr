"""Audit SFT-anchored GRPO against exact destination-only control rows."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    steps = int(sys.argv[1])
    seed = int(sys.argv[2])
    assert steps in (2, 64) and seed == 11
    root = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
    name = f'three_directions_kl_anchor_{steps}step_seed{seed}'
    candidate_root = root / 'verl_checkpoints' / name
    control_root = root / 'verl_checkpoints/three_directions_branch_control_64step'
    run_dir = root / 'runlogs' / name
    assert hashlib.sha256((root / 'data/branch_pilot_train.parquet').read_bytes()).hexdigest() == (
        'a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3')
    assert (candidate_root / f'global_step_{steps}/actor/huggingface/config.json').exists()
    config = (run_dir / 'config.txt').read_text()
    assert 'rollout_n=2 actor_kl_coef=0.001' in config
    candidate = [json.loads(line) for line in (candidate_root / 'rollout.jsonl').read_text().splitlines()]
    control = [json.loads(line) for line in (control_root / 'rollout.jsonl').read_text().splitlines()[:steps]]
    assert [r['step'] for r in candidate] == [r['step'] for r in control] == list(range(1, steps + 1))
    seen = set()
    diverse_groups = varied_groups = 0
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
        control_ids = collections.Counter(str(item['episode_id']) for item in b['info'])
        assert len(by_episode) == len(control_ids) == 4
        assert set(by_episode) == set(control_ids)
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
    log = (run_dir / 'train.log').read_text()
    grads = [float(x) for x in re.findall(r'actor/grad_norm:([0-9.eE+-]+)', log)]
    kl_losses = [float(x) for x in re.findall(r'actor/kl_loss:([0-9.eE+-]+)', log)]
    assert len(grads) >= steps and all(math.isfinite(x) and x >= 0 for x in grads[:steps])
    assert any(x > 1e-6 for x in grads[:steps])
    zero_grad_steps = [i + 1 for i, x in enumerate(grads[:steps]) if x == 0]
    assert set(zero_grad_steps).issubset(all_tied_steps)
    assert len(kl_losses) >= steps and all(math.isfinite(x) for x in kl_losses[:steps])
    event_files = list((run_dir / 'tensorboard').glob('events.out.tfevents.*'))
    assert event_files
    kl_by_step, coef_by_step = {}, {}
    for path in sorted(event_files):
        events = EventAccumulator(str(path), size_guidance={'scalars': 0})
        events.Reload()
        assert {'actor/kl_loss', 'actor/kl_coef'} <= set(events.Tags()['scalars'])
        kl_by_step.update((x.step, x.value) for x in events.Scalars('actor/kl_loss'))
        coef_by_step.update((x.step, x.value) for x in events.Scalars('actor/kl_coef'))
    tb_kl = [kl_by_step[i] for i in range(1, steps + 1)]
    tb_coef = [coef_by_step[i] for i in range(1, steps + 1)]
    assert all(math.isfinite(x) for x in tb_kl)
    assert all(math.isclose(x, 0.001, abs_tol=1e-7) for x in tb_coef)
    assert diverse_groups > 0 and varied_groups > 0
    print(json.dumps({
        'seed': seed, 'steps': steps, 'rollout_n': 2, 'actor_kl_coef': 0.001,
        'matched_train_episode_sets_at_each_step': steps,
        'unique_train_episodes': len(seen), 'candidate_rollouts': steps * 8,
        'diverse_candidate_groups': diverse_groups,
        'nonzero_return_variance_candidate_groups': varied_groups,
        'actor_grad_norms': grads[:steps], 'actor_kl_losses_console_rounded': kl_losses[:steps],
        'actor_kl_losses_tensorboard': tb_kl,
        'actor_kl_nonzero_tensorboard_steps': sum(abs(x) > 1e-9 for x in tb_kl),
        'actor_kl_coef_tensorboard': tb_coef,
        'all_tied_steps': all_tied_steps, 'zero_grad_steps': zero_grad_steps,
        'interpretation': 'KL and training-row audit, not held-out navigation evidence.',
    }, indent=2))


if __name__ == '__main__':
    main()
