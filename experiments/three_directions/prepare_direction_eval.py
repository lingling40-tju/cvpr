"""Freeze a scene-balanced val-unseen pilot and extract existing baselines."""

import collections
import json
from pathlib import Path


BASE = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930')
OUT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002')
OLD = BASE / 'runlogs/eventtrace_full_val_unseen'
RESULT = OUT / 'runlogs/three_direction_val256'
COUNT = 256


def main():
    complete = json.loads((OLD / 'manifest.json').read_text())
    ids = [str(x) for x in complete['episode_ids']]
    scenes = [str(x) for x in complete['scene_ids']]
    assert len(ids) == len(scenes) == 1839 and len(set(ids)) == 1839
    groups = collections.defaultdict(list)
    for index, (episode_id, scene) in enumerate(zip(ids, scenes)):
        groups[scene].append((episode_id, scene, index))
    selected = []
    for round_index in range(max(map(len, groups.values()))):
        for scene in sorted(groups):
            if round_index < len(groups[scene]):
                selected.append(groups[scene][round_index])
                if len(selected) == COUNT:
                    break
        if len(selected) == COUNT:
            break
    assert len(selected) == COUNT
    RESULT.mkdir(parents=True, exist_ok=True)
    manifest = {'split': 'val_unseen',
                'selection': 'scene-balanced fixed subset of the complete 1839-episode manifest',
                'episode_ids': [x[0] for x in selected],
                'scene_ids': [x[1] for x in selected]}
    path = RESULT / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    baselines = {}
    for label in ('sft', 'seed11_control', 'seed22_control', 'seed33_control'):
        records = []
        for episode_id, _, old_index in selected:
            record_path = OLD / label / f'shard_{old_index % 4:02d}' / 'log' / f'stats_{episode_id}_0.json'
            record = json.loads(record_path.read_text())
            assert str(record['id']) == episode_id
            assert record.get('early_stop_reason') != 'inference_error'
            records.append(record)
        baselines[label] = {
            'count': len(records),
            'successes': sum(bool(x['success']) for x in records),
            'sr': sum(bool(x['success']) for x in records) / COUNT,
            'spl': sum(float(x['spl']) for x in records) / COUNT,
            'mean_distance_to_goal': sum(float(x['distance_to_goal']) for x in records) / COUNT,
            'inference_errors': 0,
        }
    (RESULT / 'existing_baselines.json').write_text(json.dumps(baselines, indent=2) + '\n')
    print(json.dumps({'scenes': len(groups), 'pilot_episodes': COUNT,
                      'baselines': baselines}, indent=2))


if __name__ == '__main__':
    main()
