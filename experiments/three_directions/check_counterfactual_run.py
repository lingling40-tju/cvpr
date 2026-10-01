"""Check that counterfactual group rewards produced policy updates."""

import argparse
import collections
import json
import math
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rollout', type=Path)
    parser.add_argument('train_log', type=Path)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.rollout.read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, args.steps + 1))
    diverse = 0
    for record in records:
        assert len(record['info']) == 8
        groups = collections.defaultdict(list)
        for info in record['info']:
            assert info['global_start_step'] == 1
            groups[str(info['episode_id'])].append(info)
        assert len(groups) == 4 and all(len(pair) == 2 for pair in groups.values())
        for pair in groups.values():
            signatures = [tuple(turn['response'] for turn in info['gen_traj']) for info in pair]
            diverse += signatures[0] != signatures[1]
    log = args.train_log.read_text(errors='replace')
    grads = [float(x) for x in re.findall(r'actor/grad_norm:([0-9.eE+-]+)', log)]
    correct = [float(x) for x in re.findall(r'counterfactual/first_turn_correct:([0-9.eE+-]+)', log)]
    assert len(correct) == args.steps and all(math.isfinite(x) and 0 <= x <= 1 for x in correct)
    assert diverse > 0 and any(x > 1e-6 for x in grads)
    print(json.dumps({'steps': args.steps, 'diverse_episode_groups': diverse,
                      'actor_grad_norms': grads,
                      'first_turn_correct_rates': correct}, indent=2))


if __name__ == '__main__':
    main()
