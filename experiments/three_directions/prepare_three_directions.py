"""Create three auditable R2R pilot curricula from existing training data.

Branch and recovery prefixes come from completed seed-matched control rollouts.
Counterfactual pairs are natural R2R instructions sharing scene and start pose.
Only train-split episodes are used. The script never calls a semantic judge.
"""

import collections
import gzip
import itertools
import json
import math
import re
from pathlib import Path

import pandas as pd


BASE = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930')
OUT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
SOURCE = BASE / 'data/r2r_4000_train.parquet'
EPISODES = BASE / 'data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz'
ALLOWED = re.compile(r'^(move forward (25|50|75)cm|turn (left|right) (15|30|45) degrees)$')


def actions(info):
    result = []
    for turn in info.get('gen_traj', []):
        for action in turn.get('executed_actions', []):
            if action == 'stop':
                return result
            if not ALLOWED.fullmatch(action):
                return []
            result.append(action)
    return result


def first_turn(gt):
    degrees = 0
    for action in gt:
        if action.startswith('move forward'):
            break
        match = re.search(r'\d+', action)
        if match:
            degrees += int(match.group()) * (1 if action.startswith('turn right') else -1)
    return degrees


def copy_row(row, **updates):
    result = row.to_dict()
    result['extra_info'] = dict(result['extra_info'])
    result['extra_info'].update(updates)
    return result


def main():
    source = pd.read_parquet(SOURCE)
    # Named itertuples retain the source row index for copying.
    by_id = {str(row.extra_info['episode_id']): row for row in source.itertuples(index=True)}
    outcomes = collections.defaultdict(list)
    for seed in (11, 22, 33):
        path = BASE / f'verl_checkpoints/eventtrace_r2r64_seed{seed}_control/rollout.jsonl'
        for line in path.read_text().splitlines():
            record = json.loads(line)
            for info in record['info']:
                if info.get('global_start_step') == 1:
                    outcomes[str(info['episode_id'])].append(info)

    branch, recovery = [], []
    for episode_id, rows in outcomes.items():
        row = by_id.get(episode_id)
        if row is None or len(row.extra_info['gt_actions']) < 18:
            continue
        branch_candidates = []
        recovery_candidates = []
        gt = list(row.extra_info['gt_actions'])
        for info in rows:
            seq = actions(info)
            if len(seq) < 7:
                continue
            first_motion = next((i for i, a in enumerate(seq) if a.startswith('move forward')), None)
            if first_motion is not None:
                next_turn = next((i for i in range(first_motion + 1, min(len(seq), 12))
                                  if seq[i].startswith('turn ')), None)
                if next_turn is not None and 4 <= next_turn <= 9:
                    branch_candidates.append(seq[:next_turn])
            if (not info['task_success'] and
                    3.5 < float(info['distance_to_goal']) < 15 and len(seq) >= 9):
                prefix = seq[:9]
                if prefix[:6] != gt[:6]:
                    recovery_candidates.append(prefix)
        if branch_candidates:
            branch.append((episode_id, branch_candidates[0]))
        if recovery_candidates:
            recovery.append((episode_id, recovery_candidates[0]))

    branch = sorted(branch)[:256]
    recovery = sorted(recovery)[:256]
    if len(branch) < 256 or len(recovery) < 256:
        raise RuntimeError(f'insufficient prefixes: branch={len(branch)}, recovery={len(recovery)}')

    for mode, records in (('branch', branch), ('recovery', recovery)):
        rows = [copy_row(source.iloc[row.Index],
                         forced_history_actions=list(prefix),
                         alternative_mode=mode)
                for episode_id, prefix in records
                for row in [by_id[episode_id]]]
        pd.DataFrame(rows).to_parquet(OUT / f'data/{mode}_pilot_train.parquet', index=False)

    with gzip.open(EPISODES, 'rt') as stream:
        episodes = json.load(stream)['episodes']
    groups = collections.defaultdict(list)
    for e in episodes:
        if str(e['episode_id']) not in by_id:
            continue
        pose = (e['scene_id'],
                tuple(round(float(x), 3) for x in e['start_position']),
                tuple(round(float(x), 3) for x in e['start_rotation']))
        groups[pose].append(e)
    candidates = []
    for group in groups.values():
        for a, b in itertools.combinations(group, 2):
            if math.dist(a['goals'][0]['position'], b['goals'][0]['position']) <= 3:
                continue
            ya = first_turn(by_id[str(a['episode_id'])].extra_info['gt_actions'])
            yb = first_turn(by_id[str(b['episode_id'])].extra_info['gt_actions'])
            if ya * yb < 0 and min(abs(ya), abs(yb)) >= 30:
                candidates.append((a, b, ya, yb))
    candidates.sort(key=lambda x: (-math.dist(x[0]['goals'][0]['position'],
                                              x[1]['goals'][0]['position']),
                                   str(x[0]['episode_id']), str(x[1]['episode_id'])))
    used = set()
    disjoint = []
    for a, b, ya, yb in candidates:
        if str(a['episode_id']) in used or str(b['episode_id']) in used:
            continue
        disjoint.append((a, b, ya, yb))
        used.update((str(a['episode_id']), str(b['episode_id'])))
    if len(disjoint) < 64:
        raise RuntimeError(f'insufficient disjoint counterfactual pairs: {len(disjoint)}')
    pairs = [disjoint[i % len(disjoint)] for i in range(128)]
    cf_rows = []
    for index, (a, b, ya, yb) in enumerate(pairs):
        for episode, yaw in ((a, ya), (b, yb)):
            source_row = by_id[str(episode['episode_id'])]
            cf_rows.append(copy_row(source.iloc[source_row.Index],
                                    cf_pair_id=f'pair_{index:03d}',
                                    cf_turn_sign=1 if yaw > 0 else -1,
                                    cf_goal_position=episode['goals'][0]['position'],
                                    alternative_mode='counterfactual'))
    pd.DataFrame(cf_rows).to_parquet(OUT / 'data/counterfactual_pilot_train.parquet', index=False)

    diagnostics = {
        'source_train_rows': len(source),
        'branch_episode_count': len(branch),
        'branch_prefix_lengths': collections.Counter(len(p) for _, p in branch),
        'recovery_episode_count': len(recovery),
        'recovery_prefix_lengths': collections.Counter(len(p) for _, p in recovery),
        'natural_same_start_opposite_turn_candidates': len(candidates),
        'disjoint_counterfactual_pairs': len(disjoint),
        'counterfactual_train_pairs': len(pairs),
        'counterfactual_train_rows': len(cf_rows),
    }
    (OUT / 'runlogs/three_direction_data_diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2, default=dict) + '\n')
    print(json.dumps(diagnostics, indent=2, default=dict))


if __name__ == '__main__':
    main()
