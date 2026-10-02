"""Audit the isolated 2+2 GRPO ablation against four-way group sampling."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


DATASET_SHA = {
    2: 'a774da703ae3b94d5c138db23f07160f2f94bccceb406f87b4d2cc8d0b5655e3',
    128: '2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea',
}


def read_steps(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=(2, 128), required=True)
    parser.add_argument('--seed', type=int, choices=(11, 22, 33), required=True)
    args = parser.parse_args()
    assert args.steps == 128 or args.seed == 11
    name = f'three_directions_group4_pairwise_{args.steps}step_seed{args.seed}'
    source_name = ('three_directions_group4_64step_seed11' if args.steps == 2
                   else 'three_directions_group4_128step' +
                   ('' if args.seed == 11 else f'_seed{args.seed}'))
    checkpoint = args.root / 'verl_checkpoints' / name
    run_dir = args.root / 'runlogs' / name
    source_checkpoint = args.source_root / 'verl_checkpoints' / source_name
    assert (checkpoint / f'global_step_{args.steps}/actor/huggingface/config.json').exists()
    assert (args.source_root / 'runlogs' / source_name / 'completed').exists()
    dataset_name = ('branch_pilot_train.parquet' if args.steps == 2
                    else 'branch_scale512_train.parquet')
    dataset = args.root / 'data' / dataset_name
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == DATASET_SHA[args.steps]
    info = list(pd.read_parquet(dataset)['extra_info'])
    episode_ids = [str(x['episode_id']) for x in info]
    assert len(episode_ids) == len(set(episode_ids))
    assert all(x['split'] == 'train' for x in info)
    config = (run_dir / 'config.txt').read_text()
    assert f'steps={args.steps} seed={args.seed} ' in config
    assert f'dataset_sha256={DATASET_SHA[args.steps]}' in config
    assert 'rollout_n=4 pairwise_uid=1' in config
    candidate = read_steps(checkpoint / 'rollout.jsonl')
    reference = read_steps(source_checkpoint / 'rollout.jsonl')[:args.steps]
    assert [x['step'] for x in candidate] == [x['step'] for x in reference] == list(range(1, args.steps + 1))

    seen = set()
    varied_pairs = 0
    all_tied_steps = []
    for record, baseline in zip(candidate, reference):
        step = record['step']
        by_episode = collections.defaultdict(list)
        by_uid = collections.defaultdict(list)
        for item in record['info']:
            episode = str(item['episode_id'])
            uid = str(item['grpo_uid'])
            assert uid.endswith((':pair0', ':pair1'))
            assert item['env_global_step'] == item['env_local_step']
            assert math.isfinite(float(item['total_reward']))
            parts = item['reward_components']
            assert float(parts['ndtw_reward']) == 0
            assert float(parts['semantic_reward']) == 0
            assert 'geodesic_progress_reward' not in parts
            by_episode[episode].append(item)
            by_uid[uid].append(item)
        original_ids = collections.Counter(str(x['episode_id']) for x in baseline['info'])
        expected_ids = set(episode_ids[4 * (step - 1):4 * step])
        assert set(by_episode) == set(original_ids) == expected_ids
        assert set(original_ids.values()) == {4}
        assert len(by_episode) == 4 and set(len(x) for x in by_episode.values()) == {4}
        assert len(by_uid) == 8 and set(len(x) for x in by_uid.values()) == {2}
        for items in by_uid.values():
            assert len({str(x['episode_id']) for x in items}) == 1
        for episode, items in by_episode.items():
            uids = {str(x['grpo_uid']) for x in items}
            assert len(uids) == 2
            assert {uid.rsplit(':pair', 1)[1] for uid in uids} == {'0', '1'}
            assert len({uid.rsplit(':pair', 1)[0] for uid in uids}) == 1
        assert not seen.intersection(by_episode)
        seen.update(by_episode)
        step_varied = sum(float(items[0]['total_reward']) != float(items[1]['total_reward'])
                          for items in by_uid.values())
        varied_pairs += step_varied
        if step_varied == 0:
            all_tied_steps.append(step)
    assert seen == set(episode_ids[:4 * args.steps]) and varied_pairs > 0

    log = (run_dir / 'train.log').read_text(errors='replace')
    group_metrics = [float(x) for x in re.findall(r'training/pairwise_uid_groups:([0-9.eE+-]+)', log)]
    assert len(group_metrics) >= args.steps and group_metrics[:args.steps] == [8.0] * args.steps
    grads = [float(x) for x in re.findall(r'actor/grad_norm:([0-9.eE+-]+)', log)]
    assert len(grads) >= args.steps and all(math.isfinite(x) and x >= 0 for x in grads[:args.steps])
    assert any(x > 1e-6 for x in grads[:args.steps])
    zero_grad_steps = [i + 1 for i, x in enumerate(grads[:args.steps]) if x == 0]
    assert set(zero_grad_steps).issubset(all_tied_steps)
    print(json.dumps({
        'mode': 'group4_pairwise', 'seed': args.seed, 'steps': args.steps,
        'dataset_sha256': DATASET_SHA[args.steps],
        'matched_train_episode_sets_at_each_step': args.steps,
        'unique_train_episodes': len(seen),
        'candidate_rollouts': 16 * args.steps,
        'four_way_reference_rollouts': 16 * args.steps,
        'pairwise_uid_groups_per_step': 8,
        'nonzero_return_variance_pairs': varied_pairs,
        'all_tied_steps': all_tied_steps,
        'zero_grad_steps': zero_grad_steps,
        'actor_grad_norms': grads[:args.steps],
        'interpretation': 'Compute and pairing audit; not held-out navigation evidence.',
    }, indent=2))


if __name__ == '__main__':
    main()
