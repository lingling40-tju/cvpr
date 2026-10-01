"""Prepare 512-row, non-repeated train curricula if pilot results justify scaling.

Only R2R train episodes are used. The branch/recovery prefixes come from
completed train rollouts, while counterfactual pairs come from the full train
JSON and action supervision. Existing Parquet action grouping is preserved.
"""

import collections
import copy
import gzip
import itertools
import json
import math
import re

import pandas as pd

from prepare_three_directions import BASE, OUT, SOURCE, EPISODES, actions, copy_row


COUNT = 512
GT = EPISODES.with_name('train_gt.json.gz')
VAL_EPISODES = EPISODES.parent.parent / 'val_unseen/val_unseen.json.gz'
TEXT = {1: 'move forward', 2: 'turn left', 3: 'turn right'}
UNIT = {1: 25, 2: 15, 3: 15}


def primitive_actions(text_actions):
    result = []
    for action in text_actions:
        if action == 'stop':
            continue
        kind = 1 if action.startswith('move forward') else 2 if action.startswith('turn left') else 3
        number = re.search(r'\d+', action)
        if number is None:
            raise ValueError(f'bad action: {action}')
        count = int(number.group()) // UNIT[kind]
        if count not in (1, 2, 3):
            raise ValueError(f'bad count: {action}')
        result.extend([kind] * count)
    return result


def compress_actions(numeric):
    if not numeric or numeric[-1] != 0 or any(x not in (0, 1, 2, 3) for x in numeric):
        raise ValueError('invalid ground-truth action sequence')
    if numeric.count(0) != 1:
        raise ValueError('stop must appear exactly once, at the end')
    result = []
    steps = numeric[:-1]
    position = 0
    while position < len(steps):
        kind = steps[position]
        length = 1
        while length < 3 and position + length < len(steps) and steps[position + length] == kind:
            length += 1
        unit = 'cm' if kind == 1 else ' degrees'
        result.append(f'{TEXT[kind]} {length * UNIT[kind]}{unit}')
        position += length
    result.append('stop')
    assert primitive_actions(result) == steps
    return result


def first_turn(numeric):
    yaw = 0
    for action in numeric:
        if action in (0, 1):
            break
        yaw += 1 if action == 3 else -1
    return yaw


def main():
    source = pd.read_parquet(SOURCE)
    by_id = {str(row.extra_info['episode_id']): row for row in source.itertuples(index=True)}
    assert len(by_id) == len(source) == 4000
    with gzip.open(GT, 'rt') as stream:
        ground_truth = json.load(stream)
    with gzip.open(EPISODES, 'rt') as stream:
        episodes = json.load(stream)['episodes']
    with gzip.open(VAL_EPISODES, 'rt') as stream:
        val_episodes = json.load(stream)['episodes']
    assert len(ground_truth) == len(episodes) == 10819
    by_episode = {str(e['episode_id']): e for e in episodes}
    assert set(by_episode) == set(ground_truth)
    train_scenes = {e['scene_id'] for e in episodes}
    val_scenes = {e['scene_id'] for e in val_episodes}
    assert not train_scenes.intersection(val_scenes)

    for episode_id, row in by_id.items():
        expected = ground_truth[episode_id]['actions'][:-1]
        assert primitive_actions(row.extra_info['gt_actions']) == expected, episode_id

    outcomes = collections.defaultdict(list)
    for seed in (11, 22, 33):
        path = BASE / f'verl_checkpoints/eventtrace_r2r64_seed{seed}_control/rollout.jsonl'
        for line in path.read_text().splitlines():
            for info in json.loads(line)['info']:
                if info.get('global_start_step') == 1:
                    outcomes[str(info['episode_id'])].append(info)
    branch, recovery = [], []
    for episode_id, records in outcomes.items():
        row = by_id.get(episode_id)
        if row is None or len(row.extra_info['gt_actions']) < 18:
            continue
        gt = list(row.extra_info['gt_actions'])
        branch_prefix = recovery_prefix = None
        for info in records:
            seq = actions(info)
            if len(seq) < 7:
                continue
            first_motion = next((i for i, a in enumerate(seq) if a.startswith('move forward')), None)
            if first_motion is not None and branch_prefix is None:
                next_turn = next((i for i in range(first_motion + 1, min(len(seq), 12))
                                  if seq[i].startswith('turn ')), None)
                if next_turn is not None and 4 <= next_turn <= 9:
                    branch_prefix = seq[:next_turn]
            if (recovery_prefix is None and not info['task_success'] and
                    3.5 < float(info['distance_to_goal']) < 15 and len(seq) >= 9 and
                    seq[:6] != gt[:6]):
                recovery_prefix = seq[:9]
        if branch_prefix is not None:
            branch.append((episode_id, branch_prefix))
        if recovery_prefix is not None:
            recovery.append((episode_id, recovery_prefix))
    branch.sort()
    recovery.sort()
    assert len(branch) >= COUNT and len(recovery) >= COUNT

    for mode, records in (('branch', branch), ('recovery', recovery)):
        rows = [copy_row(source.iloc[by_id[eid].Index],
                         forced_history_actions=list(prefix),
                         alternative_mode=mode)
                for eid, prefix in records[:COUNT]]
        pd.DataFrame(rows).to_parquet(OUT / f'data/{mode}_scale512_train.parquet', index=False)

    groups = collections.defaultdict(list)
    for episode in episodes:
        pose = (episode['scene_id'],
                tuple(round(float(x), 3) for x in episode['start_position']),
                tuple(round(float(x), 3) for x in episode['start_rotation']))
        groups[pose].append(episode)
    candidates = []
    for group in groups.values():
        for a, b in itertools.combinations(group, 2):
            distance = math.dist(a['goals'][0]['position'], b['goals'][0]['position'])
            if distance <= 3:
                continue
            ya = first_turn(ground_truth[str(a['episode_id'])]['actions'])
            yb = first_turn(ground_truth[str(b['episode_id'])]['actions'])
            if ya * yb < 0 and min(abs(ya), abs(yb)) >= 2:
                candidates.append((distance, a, b, ya, yb))
    candidates.sort(key=lambda x: (-x[0], str(x[1]['episode_id']), str(x[2]['episode_id'])))
    used, pairs = set(), []
    for _, a, b, ya, yb in candidates:
        ia, ib = str(a['episode_id']), str(b['episode_id'])
        if ia in used or ib in used:
            continue
        pairs.append((a, b, ya, yb))
        used.update((ia, ib))
        if len(pairs) == COUNT // 2:
            break
    assert len(pairs) == COUNT // 2 and len(used) == COUNT

    template = source.iloc[0].to_dict()
    cf_rows = []
    new_episodes = 0
    for index, (a, b, ya, yb) in enumerate(pairs):
        for episode, yaw in ((a, ya), (b, yb)):
            episode_id = str(episode['episode_id'])
            if episode_id in by_id:
                row = source.iloc[by_id[episode_id].Index].to_dict()
                extra = dict(row['extra_info'])
                extra['gt_actions'] = list(extra['gt_actions'])
            else:
                row = copy.deepcopy(template)
                extra = {'episode_id': int(episode_id),
                         'gt_actions': compress_actions(ground_truth[episode_id]['actions']),
                         'split': 'train'}
                new_episodes += 1
            extra.update(cf_pair_id=f'scale_pair_{index:03d}',
                         cf_turn_sign=1 if yaw > 0 else -1,
                         cf_goal_position=episode['goals'][0]['position'],
                         alternative_mode='counterfactual')
            row['extra_info'] = extra
            cf_rows.append(row)
    assert len(cf_rows) == len({str(x['extra_info']['episode_id']) for x in cf_rows}) == COUNT
    pd.DataFrame(cf_rows).to_parquet(OUT / 'data/counterfactual_scale512_train.parquet', index=False)

    diagnostics = {
        'source_parquet_rows': len(source), 'full_train_episodes': len(episodes),
        'train_scenes': len(train_scenes), 'val_unseen_scenes': len(val_scenes),
        'train_val_unseen_scene_overlap': len(train_scenes.intersection(val_scenes)),
        'available_branch_unique_episodes': len(branch),
        'available_recovery_unique_episodes': len(recovery),
        'scale_branch_rows': COUNT, 'scale_recovery_rows': COUNT,
        'counterfactual_candidate_pairs': len(candidates),
        'scale_counterfactual_disjoint_pairs': len(pairs),
        'scale_counterfactual_rows': len(cf_rows),
        'scale_counterfactual_newly_encoded_episodes': new_episodes,
    }
    (OUT / 'runlogs/three_direction_scale512_diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2) + '\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
