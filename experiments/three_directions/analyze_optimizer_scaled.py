"""Aggregate checked three-seed optimizer comparisons on one val-unseen set."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('group4', 'kl_anchor', 'dynamic'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--count', type=int, choices=(256, 1839), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    actual_sha = hashlib.sha256((args.root / 'manifest.json').read_bytes()).hexdigest()
    assert actual_sha == MANIFEST_SHA[args.count]
    pairs = []
    for seed in (11, 22, 33):
        label = f'{args.mode}_128_seed{seed}'
        control = f'branch_control128_seed{seed}'
        assert (args.root / f'{label}.completed').exists()
        assert (args.root / f'{control}.completed').exists()
        pair_path = args.root / f'paired_{label}_vs_{control}.json'
        pair = json.loads(pair_path.read_text())
        assert pair['split'] == 'val_unseen' and pair['episodes'] == args.count
        assert pair['manifest_sha256'] == actual_sha
        assert pair['candidate'] == label and pair['control'] == control
        for arm in ('candidate_metrics', 'control_metrics'):
            assert pair[arm]['count'] == args.count
            assert pair[arm]['inference_errors'] == 0
        assert all(math.isfinite(pair['paired'][key]) for key in ('sr_pp', 'spl_pp'))
        pairs.append({'seed': seed, 'candidate': label, 'control': control,
                      'candidate_metrics': pair['candidate_metrics'],
                      'control_metrics': pair['control_metrics'],
                      'paired': pair['paired']})
    result = {'mode': args.mode, 'split': 'val_unseen', 'episodes_per_seed': args.count,
              'manifest_sha256': actual_sha, 'seeds': [11, 22, 33], 'pairs': pairs,
              'summary': {},
              'interpretation': 'Primary navigation comparison only for count 1839. '
                                'Three-seed mean and SD are descriptive, not a confidence interval.'}
    for metric in ('sr_pp', 'spl_pp'):
        values = [p['paired'][metric] for p in pairs]
        result['summary'][metric] = {'values': values, 'mean': statistics.mean(values),
                                     'sample_sd': statistics.stdev(values)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
