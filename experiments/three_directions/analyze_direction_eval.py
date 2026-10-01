"""Validate a fixed val-unseen pilot and compare each model episode by episode."""

import argparse
import json
import random
from pathlib import Path


OLD = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930/runlogs/eventtrace_full_val_unseen')
ROOT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002/runlogs/three_direction_val256')


def summarize(rows, ids):
    n = len(ids)
    return {
        'count': n,
        'successes': sum(bool(rows[i]['success']) for i in ids),
        'sr': sum(bool(rows[i]['success']) for i in ids) / n,
        'spl': sum(float(rows[i]['spl']) for i in ids) / n,
        'mean_distance_to_goal': sum(float(rows[i]['distance_to_goal']) for i in ids) / n,
        'inference_errors': sum(rows[i].get('early_stop_reason') == 'inference_error' for i in ids),
    }


def scene_bootstrap(arm, base, ids, scenes, draws=10000):
    """Paired interval with scene, rather than episode, as the resampling unit."""
    scene_to_ids = {}
    for eid, scene in zip(ids, scenes):
        scene_to_ids.setdefault(scene, []).append(eid)
    scene_names = sorted(scene_to_ids)
    rng = random.Random(20261002)
    sr_draws, spl_draws = [], []
    for _ in range(draws):
        sample = [eid for _ in scene_names
                  for eid in scene_to_ids[rng.choice(scene_names)]]
        sr_draws.append(100 * sum(bool(arm[i]['success']) - bool(base[i]['success'])
                                  for i in sample) / len(sample))
        spl_draws.append(100 * sum(float(arm[i]['spl']) - float(base[i]['spl'])
                                   for i in sample) / len(sample))
    sr_draws.sort()
    spl_draws.sort()
    lo, hi = int(draws * .025), int(draws * .975)
    return {'resampling_unit': 'scene', 'scenes': len(scene_names),
            'draws': draws, 'seed': 20261002,
            'sr_pp_95': [sr_draws[lo], sr_draws[hi]],
            'spl_pp_95': [spl_draws[lo], spl_draws[hi]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', nargs='+')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    ids = [str(x) for x in manifest['episode_ids']]
    assert len(ids) == len(set(ids)) == 256
    scenes = manifest['scene_ids']
    assert len(scenes) == len(ids)
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    old_index = {str(x): i for i, x in enumerate(old_manifest['episode_ids'])}
    assert set(ids) <= set(old_index)

    rows = {}
    for label in ('sft', 'seed11_control', 'seed22_control', 'seed33_control'):
        rows[label] = {}
        for eid in ids:
            path = OLD / label / f'shard_{old_index[eid] % 4:02d}/log/stats_{eid}_0.json'
            rows[label][eid] = json.loads(path.read_text())
    for label in args.labels:
        assert (ROOT / f'{label}.completed').exists(), label
        rows[label] = {}
        for shard in range(4):
            folder = ROOT / label / f'shard_{shard:02d}'
            summary = json.loads((folder / 'summary.json').read_text())
            expected = ids[shard::4]
            actual = [str(x) for x in summary['episode_ids']]
            assert set(actual) == set(expected) and len(actual) == len(expected)
            assert summary['inference_errors'] == 0
            for eid in expected:
                path = folder / 'log' / f'stats_{eid}_0.json'
                rows[label][eid] = json.loads(path.read_text())
        assert set(rows[label]) == set(ids)

    output = {'split': 'val_unseen', 'episode_count': 256,
              'scene_count': len(set(manifest['scene_ids'])),
              'models': {label: summarize(value, ids) for label, value in rows.items()},
              'paired_vs_seed11_control': {}}
    for label in args.labels:
        arm = rows[label]
        base = rows['seed11_control']
        output['paired_vs_seed11_control'][label] = {
            'sr_pp': 100 * sum(bool(arm[i]['success']) - bool(base[i]['success']) for i in ids) / len(ids),
            'spl_pp': 100 * sum(float(arm[i]['spl']) - float(base[i]['spl']) for i in ids) / len(ids),
            'candidate_only_successes': sum(bool(arm[i]['success']) and not bool(base[i]['success']) for i in ids),
            'baseline_only_successes': sum(bool(base[i]['success']) and not bool(arm[i]['success']) for i in ids),
            'scene_cluster_bootstrap95': scene_bootstrap(arm, base, ids, scenes),
        }
    (ROOT / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
