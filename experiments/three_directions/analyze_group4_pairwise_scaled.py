"""Aggregate matched 2+2 GRPO comparisons across three training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


MANIFEST_SHA = {
    256: '546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46',
    1839: '262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e',
}
SEEDS = (11, 22, 33)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--count', type=int, choices=(256, 1839), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_sha = hashlib.sha256((args.root / 'manifest.json').read_bytes()).hexdigest()
    assert manifest_sha == MANIFEST_SHA[args.count]
    result = {
        'mode': 'group4_pairwise', 'split': 'val_unseen',
        'episodes_per_seed': args.count, 'manifest_sha256': manifest_sha,
        'seeds': list(SEEDS), 'comparisons': {},
        'interpretation': 'The complete 1,839 episodes are primary; three-seed '
                          'mean and sample SD are descriptive, not a confidence interval.',
    }
    candidate_by_seed = {}
    for comparison in ('branch_control', 'group4'):
        pairs = []
        for seed in SEEDS:
            label = f'group4_pairwise_128_seed{seed}'
            control = (f'branch_control128_seed{seed}' if comparison == 'branch_control'
                       else f'group4_128_seed{seed}')
            assert (args.root / f'{label}.completed').exists()
            assert (args.root / f'{control}.completed').exists()
            path = args.root / f'paired_{label}_vs_{control}.json'
            pair = json.loads(path.read_text())
            assert pair['split'] == 'val_unseen' and pair['episodes'] == args.count
            assert pair['manifest_sha256'] == manifest_sha
            assert pair['candidate'] == label and pair['control'] == control
            for arm in ('candidate_metrics', 'control_metrics'):
                assert pair[arm]['count'] == args.count
                assert pair[arm]['inference_errors'] == 0
            assert all(math.isfinite(float(pair['paired'][metric]))
                       for metric in ('sr_pp', 'spl_pp'))
            candidate_metrics = pair['candidate_metrics']
            if seed in candidate_by_seed:
                assert candidate_metrics == candidate_by_seed[seed]
            else:
                candidate_by_seed[seed] = candidate_metrics
            pairs.append({
                'seed': seed, 'candidate': label, 'control': control,
                'candidate_metrics': candidate_metrics,
                'control_metrics': pair['control_metrics'],
                'paired': pair['paired'],
            })
        summary = {}
        for metric in ('sr_pp', 'spl_pp'):
            values = [p['paired'][metric] for p in pairs]
            summary[metric] = {
                'values': values, 'mean': statistics.mean(values),
                'sample_sd': statistics.stdev(values),
            }
        result['comparisons'][comparison] = {'pairs': pairs, 'summary': summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({comparison: entry['summary']
                      for comparison, entry in result['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    main()
