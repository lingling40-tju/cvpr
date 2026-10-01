"""Score instruction-dependent initial turns on frozen natural val-unseen pairs."""

import argparse
import json
import random
import re
from pathlib import Path


ROOT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002/runlogs/three_direction_val_counterfactual')
OLD = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930/runlogs/eventtrace_full_val_unseen')
BASELINES = ('sft', 'seed11_control', 'seed22_control', 'seed33_control')


def first_turn_sign(record):
    yaw = 0
    for turn in record.get('conversations', []):
        if turn.get('role') != 'assistant':
            continue
        for part in turn.get('content', []):
            if part.get('type') != 'text':
                continue
            for action in part.get('text', '').lower().split(','):
                if 'stop' in action or 'forward' in action:
                    return (yaw > 0) - (yaw < 0)
                if 'left' in action or 'right' in action:
                    match = re.search(r'\d+', action)
                    degrees = int(match.group()) if match else 15
                    yaw += degrees if 'right' in action else -degrees
    return (yaw > 0) - (yaw < 0)


def load_old(label, ids):
    assert (OLD / f'{label}.completed').exists()
    old_ids = [str(x) for x in json.loads((OLD / 'manifest.json').read_text())['episode_ids']]
    old_index = {eid: i for i, eid in enumerate(old_ids)}
    assert set(ids) <= set(old_index)
    result = {}
    for eid in ids:
        shard = old_index[eid] % 4
        folder = OLD / label / f'shard_{shard:02d}'
        result[eid] = first_turn_sign(json.loads(
            (folder / 'extra_info' / f'info_{eid}_0.json').read_text()))
    return result


def load_candidate(label, ids):
    assert (ROOT / f'{label}.completed').exists()
    result = {}
    for shard in range(4):
        folder = ROOT / label / f'shard_{shard:02d}'
        summary = json.loads((folder / 'summary.json').read_text())
        expected = ids[shard::4]
        observed_ids = [str(x) for x in summary['episode_ids']]
        assert len(observed_ids) == len(set(observed_ids)) == len(expected)
        assert set(observed_ids) == set(expected)
        assert summary['count'] == len(expected)
        assert summary['inference_errors'] == 0
        for eid in expected:
            result[eid] = first_turn_sign(json.loads(
                (folder / 'extra_info' / f'info_{eid}_0.json').read_text()))
    assert set(result) == set(ids)
    return result


def score(observed, pairs):
    episode_correct = both_correct = responsive = 0
    for pair in pairs:
        a, b = pair['episode_ids']
        expected_a, expected_b = pair['expert_initial_turn_signs']
        got_a, got_b = observed[a], observed[b]
        episode_correct += got_a == expected_a
        episode_correct += got_b == expected_b
        both_correct += got_a == expected_a and got_b == expected_b
        responsive += got_a * got_b == -1
    n = len(pairs)
    return {'pairs': n, 'episodes': n * 2,
            'episode_expert_turn_agreement': episode_correct / (n * 2),
            'both_expert_turn_agreement_rate': both_correct / n,
            'opposite_turn_pair_rate': responsive / n,
            'no_initial_turn_episodes': sum(observed[eid] == 0 for p in pairs for eid in p['episode_ids'])}


def paired_bootstrap(arm, baseline, pairs, draws=10000):
    rng = random.Random(20261002)
    scene_pairs = {}
    for pair in pairs:
        scene_pairs.setdefault(pair['scene_id'], []).append(pair)
    scenes = sorted(scene_pairs)
    values = []
    for _ in range(draws):
        sample = [pair for _ in scenes for pair in scene_pairs[rng.choice(scenes)]]
        values.append(score(arm, sample)['episode_expert_turn_agreement'] -
                      score(baseline, sample)['episode_expert_turn_agreement'])
    values.sort()
    return {'resampling_unit': 'scene', 'scenes': len(scenes),
            'draws': draws, 'seed': 20261002,
            'episode_expert_turn_agreement_pp_95': [100 * values[int(draws * .025)],
                                                    100 * values[int(draws * .975)]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', help='completed candidate label in this manifest')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    pairs = manifest['pairs']
    ids = [str(x) for x in manifest['episode_ids']]
    assert len(ids) == len(set(ids)) == 2 * len(pairs)
    assert all(pair['expert_initial_turn_signs'] in ([1, -1], [-1, 1]) for pair in pairs)
    observed = {label: load_old(label, ids) for label in BASELINES}
    if args.candidate:
        observed[args.candidate] = load_candidate(args.candidate, ids)
    output = {'split': 'val_unseen', 'natural_disjoint_pairs': len(pairs),
              'scenes': len(set(manifest['scene_ids'])),
              'models': {label: score(signs, pairs) for label, signs in observed.items()},
              'paired_vs_sft': {}, 'per_pair': []}
    if args.candidate:
        arm = observed[args.candidate]
        for baseline_name in ('sft', 'seed11_control'):
            base = observed[baseline_name]
            output['paired_vs_sft' if baseline_name == 'sft' else 'paired_vs_seed11_control'] = {
                args.candidate: {
                    'episode_expert_turn_agreement_pp': 100 * (score(arm, pairs)['episode_expert_turn_agreement'] -
                                                              score(base, pairs)['episode_expert_turn_agreement']),
                    'scene_cluster_bootstrap95': paired_bootstrap(arm, base, pairs),
                }
            }
    for pair in pairs:
        output['per_pair'].append({
            'episode_ids': pair['episode_ids'], 'scene_id': pair['scene_id'],
            'expert_signs': pair['expert_initial_turn_signs'],
            'observed_signs': {label: [signs[eid] for eid in pair['episode_ids']]
                               for label, signs in observed.items()},
        })
    path = ROOT / ('analysis.json' if args.candidate else 'baseline_analysis.json')
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'per_pair'}, indent=2))


if __name__ == '__main__':
    main()
