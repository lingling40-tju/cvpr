"""Freeze natural, disjoint same-start instruction pairs from val-unseen."""

import collections
import gzip
import itertools
import json
import math
from pathlib import Path


ROOT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
DATA = ROOT / 'data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen'
OUT = ROOT / 'runlogs/three_direction_val_counterfactual'


def first_turn(actions):
    yaw = 0
    for action in actions:
        if action in (0, 1):
            break
        yaw += 1 if action == 3 else -1
    return yaw


def main():
    with gzip.open(DATA / 'val_unseen.json.gz', 'rt') as stream:
        episodes = json.load(stream)['episodes']
    with gzip.open(DATA / 'val_unseen_gt.json.gz', 'rt') as stream:
        ground_truth = json.load(stream)
    assert len(episodes) == len(ground_truth) == 1839
    groups = collections.defaultdict(list)
    for episode in episodes:
        episode_id = str(episode['episode_id'])
        yaw = first_turn(ground_truth[episode_id]['actions'])
        if abs(yaw) < 2:
            continue
        pose = (episode['scene_id'],
                tuple(round(float(x), 3) for x in episode['start_position']),
                tuple(round(float(x), 3) for x in episode['start_rotation']))
        groups[pose].append((episode, yaw))
    candidates = []
    for group in groups.values():
        for (a, ya), (b, yb) in itertools.combinations(group, 2):
            distance = math.dist(a['goals'][0]['position'], b['goals'][0]['position'])
            if distance > 3 and ya * yb < 0:
                candidates.append((distance, a, b, ya, yb))
    candidates.sort(key=lambda x: (-x[0], str(x[1]['episode_id']), str(x[2]['episode_id'])))
    used, pairs = set(), []
    for distance, a, b, ya, yb in candidates:
        ids = (str(a['episode_id']), str(b['episode_id']))
        if ids[0] in used or ids[1] in used:
            continue
        used.update(ids)
        pairs.append({'episode_ids': list(ids), 'scene_id': a['scene_id'],
                      'expert_initial_turn_signs': [1 if ya > 0 else -1,
                                                    1 if yb > 0 else -1],
                      'goal_separation_m': distance})
    assert len(pairs) >= 50 and len(used) == 2 * len(pairs)
    manifest = {
        'split': 'val_unseen',
        'selection': 'all greedily disjoint natural same-start pairs, goal separation >3m, opposite >=30-degree expert initial turns; sorted by goal separation',
        'episode_ids': [eid for pair in pairs for eid in pair['episode_ids']],
        'scene_ids': [pair['scene_id'] for pair in pairs for _ in range(2)],
        'pairs': pairs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'candidate_pairs': len(candidates),
                      'disjoint_pairs': len(pairs),
                      'heldout_episodes': len(manifest['episode_ids']),
                      'scenes': len(set(manifest['scene_ids']))}, indent=2))


if __name__ == '__main__':
    main()
