"""Audit three matched seeds on a fixed val-unseen manifest."""

import argparse
import json
import random
import statistics
from pathlib import Path

from analyze_direction_eval import scene_bootstrap, summarize


ROOT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002/runlogs/three_direction_val256')


def scene_seed_bootstrap(paired_rows, ids, scenes, draws=10000):
    """Exploratory paired interval over both training seeds and held-out scenes."""
    assert len(ids) == len(scenes) and len(paired_rows) == 3
    by_scene = {}
    for eid, scene in zip(ids, scenes):
        by_scene.setdefault(scene, []).append(eid)
    scene_names = sorted(by_scene)
    seeds = sorted(paired_rows)
    deltas = {
        seed: {
            eid: (
                bool(arm[eid]['success']) - bool(control[eid]['success']),
                float(arm[eid]['spl']) - float(control[eid]['spl']),
            )
            for eid in ids
        }
        for seed, (arm, control) in paired_rows.items()
    }
    rng = random.Random(20261003)
    sampled_sr, sampled_spl = [], []
    for _ in range(draws):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_ids = [eid for _ in scene_names
                       for eid in by_scene[rng.choice(scene_names)]]
        denominator = len(sampled_seeds) * len(sampled_ids)
        sampled_sr.append(100 * sum(deltas[seed][eid][0] for seed in sampled_seeds
                                    for eid in sampled_ids) / denominator)
        sampled_spl.append(100 * sum(deltas[seed][eid][1] for seed in sampled_seeds
                                     for eid in sampled_ids) / denominator)
    sampled_sr.sort()
    sampled_spl.sort()
    lo, hi = int(draws * .025), int(draws * .975)
    return {
        'resampling_units': ['training_seed', 'held_out_scene'],
        'training_seeds': len(seeds), 'held_out_scenes': len(scene_names),
        'draws': draws, 'seed': 20261003,
        'interpretation': 'Exploratory interval; only three training seeds are available.',
        'mean_paired_sr_pp_95': [sampled_sr[lo], sampled_sr[hi]],
        'mean_paired_spl_pp_95': [sampled_spl[lo], sampled_spl[hi]],
    }


def load_label(root, label, ids):
    assert (root / f'{label}.completed').exists(), label
    rows = {}
    for shard in range(4):
        folder = root / label / f'shard_{shard:02d}'
        summary = json.loads((folder / 'summary.json').read_text())
        expected = ids[shard::4]
        observed = [str(x) for x in summary['episode_ids']]
        assert len(observed) == len(set(observed)) == len(expected)
        assert set(observed) == set(expected)
        assert summary['count'] == len(expected) and summary['inference_errors'] == 0
        for eid in expected:
            row = json.loads((folder / 'log' / f'stats_{eid}_0.json').read_text())
            assert str(row['id']) == eid and row.get('early_stop_reason') != 'inference_error'
            rows[eid] = row
    assert set(rows) == set(ids)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('branch', 'recovery', 'counterfactual'))
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--expected-count', type=int, default=256)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    ids = [str(x) for x in manifest['episode_ids']]
    scenes = manifest['scene_ids']
    assert len(ids) == len(set(ids)) == len(scenes) == args.expected_count
    output = {
        'split': 'val_unseen', 'episodes': args.expected_count, 'scenes': len(set(scenes)),
        'mode': args.mode, 'train_steps': args.steps,
        'train_rows_per_arm': 512, 'rollouts_per_episode': 2,
        'models': {}, 'paired_seed_differences': {},
    }
    paired_rows = {}
    for seed in (11, 22, 33):
        candidate_label = f'{args.mode}{args.steps}_seed{seed}'
        control_label = f'{args.mode}_control{args.steps}_seed{seed}'
        candidate = load_label(root, candidate_label, ids)
        control = load_label(root, control_label, ids)
        paired_rows[seed] = (candidate, control)
        output['models'][candidate_label] = summarize(candidate, ids)
        output['models'][control_label] = summarize(control, ids)
        result = {
            'candidate_label': candidate_label,
            'control_label': control_label,
            'sr_pp': 100 * sum(bool(candidate[i]['success']) - bool(control[i]['success'])
                               for i in ids) / len(ids),
            'spl_pp': 100 * sum(float(candidate[i]['spl']) - float(control[i]['spl'])
                                for i in ids) / len(ids),
            'candidate_only_successes': sum(bool(candidate[i]['success']) and not bool(control[i]['success']) for i in ids),
            'control_only_successes': sum(bool(control[i]['success']) and not bool(candidate[i]['success']) for i in ids),
            'scene_cluster_bootstrap95': scene_bootstrap(candidate, control, ids, scenes),
        }
        output['paired_seed_differences'][str(seed)] = result
    for metric in ('sr_pp', 'spl_pp'):
        values = [output['paired_seed_differences'][str(seed)][metric] for seed in (11, 22, 33)]
        output[f'mean_paired_{metric}'] = statistics.mean(values)
        output[f'sd_paired_{metric}'] = statistics.stdev(values)
    output['scene_seed_bootstrap95'] = scene_seed_bootstrap(paired_rows, ids, scenes)
    path = root / f'scale_{args.mode}_{args.steps}_analysis.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
