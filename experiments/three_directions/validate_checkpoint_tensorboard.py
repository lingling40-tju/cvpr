"""Validate a saved 64-step pilot when its wrapper's stdout log was lost.

This reads the original rollout records, TensorBoard scalar events, and
checkpoint files. It does not infer results from a truncated stdout log.
"""

import argparse
import collections
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('branch', 'recovery'))
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    experiment = f'three_directions_{args.mode}_{args.steps}step'
    run = ROOT / 'runlogs' / experiment
    checkpoint = ROOT / 'verl_checkpoints' / experiment
    records = [json.loads(line) for line in (checkpoint / 'rollout.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, args.steps + 1))

    groups = diverse = varied = 0
    for record in records:
        by_episode = collections.defaultdict(list)
        for item in record['info']:
            by_episode[str(item['episode_id'])].append(item)
        assert len(by_episode) == 4
        for pair in by_episode.values():
            assert len(pair) == 2
            groups += 1
            signatures = [tuple(turn['response'] for turn in item['gen_traj']) for item in pair]
            diverse += signatures[0] != signatures[1]
            varied += pair[0]['total_reward'] != pair[1]['total_reward']
    assert groups == args.steps * 4 and diverse > 0 and varied > 0

    event_files = list((run / 'tensorboard').glob('events.out.*'))
    assert len(event_files) == 1
    events = EventAccumulator(str(event_files[0]))
    events.Reload()
    steps = events.Scalars('training/global_step')
    grads = events.Scalars('actor/grad_norm')
    assert [item.step for item in steps] == list(range(1, args.steps + 1))
    assert [item.step for item in grads] == list(range(1, args.steps + 1))
    assert all(item.value == item.step for item in steps)
    assert all(math.isfinite(item.value) and item.value >= 0 for item in grads)
    assert any(item.value > 1e-6 for item in grads)

    saved = checkpoint / f'global_step_{args.steps}'
    hf = saved / 'actor/huggingface'
    assert (saved / 'data.pt').stat().st_size > 0
    assert (hf / 'config.json').stat().st_size > 0
    index = json.loads((hf / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    assert shards and all((hf / name).stat().st_size > 0 for name in shards)

    result = {
        'mode': args.mode, 'steps': args.steps, 'rollout_records': len(records),
        'groups': groups, 'diverse_trajectory_groups': diverse,
        'nonzero_return_variance_groups': varied,
        'actor_grad_event_count': len(grads),
        'nonzero_actor_grad_steps': sum(item.value > 1e-6 for item in grads),
        'final_actor_grad_norm': grads[-1].value,
        'checkpoint_model_shards': shards,
        'evidence': ['rollout.jsonl', 'tensorboard scalar events',
                     f'global_step_{args.steps}/actor/huggingface'],
        'wrapper_exit': int((run / 'failed').read_text().strip()),
        'limitation': 'stdout train.log was overwritten by a shell parse error after model save',
    }
    assert result['wrapper_exit'] == 127
    path = run / 'checkpoint_validated.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
