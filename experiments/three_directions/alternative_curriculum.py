"""Training-only helpers for three R2R navigation pilots.

The branch and recovery pilots replay an exact executed action prefix in the
simulator. The counterfactual pilot compares naturally paired instructions
from the same scene and start pose using training-only expert first turns.
"""

import collections

import torch


def forced_prefix(extra_info, mode):
    if mode not in ('branch', 'recovery'):
        return None
    prefix = [str(action) for action in extra_info['forced_history_actions']]
    if not 4 <= len(prefix) <= 9 or 'stop' in prefix:
        raise ValueError(f'invalid {mode} prefix: {prefix}')
    return prefix


def first_turn_sign(info):
    """Direction of all rotations before the first forward or stop action."""
    net = 0
    for turn in info.get('gen_traj', []):
        for action in turn.get('executed_actions', []):
            if action.startswith('move forward') or action == 'stop':
                return (net > 0) - (net < 0)
            if action.startswith('turn right'):
                net += int(action.split()[2])
            elif action.startswith('turn left'):
                net -= int(action.split()[2])
    return (net > 0) - (net < 0)


def add_counterfactual_reward(batch, reward_tensor, bonus=2.0):
    infos = batch.non_tensor_batch['info']
    metadata = batch.non_tensor_batch['extra_info']
    mask = batch.batch['response_mask']
    assert len(infos) == len(metadata) == len(reward_tensor)
    pair_counts = collections.Counter(str(x['cf_pair_id']) for x in metadata)
    assert pair_counts and set(pair_counts.values()) == {4}, pair_counts
    correct = wrong = no_turn = 0
    for i, (info, extra) in enumerate(zip(infos, metadata)):
        expected = int(extra['cf_turn_sign'])
        observed = first_turn_sign(info)
        if expected not in (-1, 1):
            raise ValueError(f'invalid expected turn: {expected}')
        if observed == expected:
            amount = bonus
            correct += 1
        elif observed == -expected:
            amount = -bonus
            wrong += 1
        else:
            amount = 0.0
            no_turn += 1
        valid = torch.nonzero(mask[i], as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError('empty response mask')
        reward_tensor[i, valid[-1]] += amount
    metrics = {
        'counterfactual/first_turn_correct': correct / len(infos),
        'counterfactual/first_turn_wrong': wrong / len(infos),
        'counterfactual/first_turn_absent': no_turn / len(infos),
    }
    return reward_tensor, metrics
