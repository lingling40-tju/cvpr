"""Recompute both compute-matched comparisons from archived paired episodes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from verify_optimizer_scaled_package import (
    MANIFEST_SHA, close, difference, digest, metrics, read_rows,
)


SEEDS = (11, 22, 33)
DATASET_SHA = '2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea'


def verify_count(root: Path, count: int) -> tuple[dict, list[str], list[str]]:
    prefix = 'val256' if count == 256 else 'full1839'
    manifest_path = root / f'{prefix}_manifest.json'
    assert digest(manifest_path) == MANIFEST_SHA[count]
    manifest = json.loads(manifest_path.read_text())
    ids = [str(x) for x in manifest['episode_ids']]
    scenes = [str(x) for x in manifest['scene_ids']]
    assert len(ids) == len(set(ids)) == len(scenes) == count
    analysis = json.loads((root / f'{prefix}_analysis.json').read_text())
    assert analysis['mode'] == 'group4_pairwise'
    assert analysis['split'] == 'val_unseen'
    assert analysis['episodes_per_seed'] == count
    assert analysis['manifest_sha256'] == MANIFEST_SHA[count]
    assert analysis['seeds'] == list(SEEDS)
    assert set(analysis['comparisons']) == {'branch_control', 'group4'}
    all_rows = {}
    for comparison in ('branch_control', 'group4'):
        entry = analysis['comparisons'][comparison]
        assert len(entry['pairs']) == 3
        rows_by_seed = {}
        for seed, pair in zip(SEEDS, entry['pairs']):
            candidate = f'group4_pairwise_128_seed{seed}'
            control = (f'branch_control128_seed{seed}' if comparison == 'branch_control'
                       else f'group4_128_seed{seed}')
            assert pair['seed'] == seed
            assert pair['candidate'] == candidate and pair['control'] == control
            rows = read_rows(root / f'paired_{count}_{comparison}_seed{seed}.jsonl',
                             ids, scenes)
            rows_by_seed[seed] = rows
            for arm in ('candidate', 'control'):
                observed = metrics(rows, ids, arm)
                expected = pair[f'{arm}_metrics']
                assert expected['inference_errors'] == 0
                assert observed['count'] == expected['count'] == count
                assert observed['successes'] == expected['successes']
                for key in ('sr', 'spl', 'mean_distance_to_goal'):
                    close(observed[key], expected[key])
            observed = difference(rows, ids)
            for key, value in observed.items():
                if key.endswith('successes'):
                    assert value == pair['paired'][key]
                else:
                    close(value, pair['paired'][key])
        for metric in ('sr_pp', 'spl_pp'):
            values = [difference(rows_by_seed[seed], ids)[metric] for seed in SEEDS]
            summary = entry['summary'][metric]
            for value, expected in zip(values, summary['values']):
                close(value, expected)
            close(statistics.mean(values), summary['mean'])
            close(statistics.stdev(values), summary['sample_sd'])
        all_rows[comparison] = rows_by_seed
        print(f'{comparison} {count}: paired SR {entry["summary"]["sr_pp"]["mean"]:+.6f} pp, '
              f'SPL {entry["summary"]["spl_pp"]["mean"]:+.6f} pp')
    for seed in SEEDS:
        control_rows = all_rows['branch_control'][seed]
        group4_rows = all_rows['group4'][seed]
        assert all(control_rows[eid]['candidate'] == group4_rows[eid]['candidate']
                   for eid in ids)
    return all_rows, ids, scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('package_dir', type=Path)
    args = parser.parse_args()
    root = args.package_dir
    filenames = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert filename not in filenames
        filenames.add(filename)
        assert digest(root / filename) == expected, filename
    required = {f'paired_{count}_{comparison}_seed{seed}.jsonl'
                for count in (256, 1839)
                for comparison in ('branch_control', 'group4')
                for seed in SEEDS}
    required |= {'val256_manifest.json', 'full1839_manifest.json',
                 'val256_analysis.json', 'full1839_analysis.json'}
    required |= {f'seed{seed}_train_audit.json' for seed in SEEDS}
    assert filenames == required
    for seed in SEEDS:
        audit = json.loads((root / f'seed{seed}_train_audit.json').read_text())
        assert audit['mode'] == 'group4_pairwise' and audit['seed'] == seed
        assert audit['steps'] == 128 and audit['dataset_sha256'] == DATASET_SHA
        assert audit['unique_train_episodes'] == 512
        assert audit['matched_train_episode_sets_at_each_step'] == 128
        assert audit['candidate_rollouts'] == audit['four_way_reference_rollouts'] == 2048
        assert audit['pairwise_uid_groups_per_step'] == 8
        assert audit['nonzero_return_variance_pairs'] > 0
        grads = audit['actor_grad_norms']
        assert len(grads) == 128 and all(math.isfinite(x) and x >= 0 for x in grads)
    screen, screen_ids, screen_scenes = verify_count(root, 256)
    full, full_ids, full_scenes = verify_count(root, 1839)
    scene_by_id = dict(zip(full_ids, full_scenes))
    assert set(screen_ids) < set(full_ids)
    assert all(scene_by_id[eid] == scene for eid, scene in zip(screen_ids, screen_scenes))
    outside = [eid for eid in full_ids if eid not in set(screen_ids)]
    assert len(outside) == 1583
    for comparison in ('branch_control', 'group4'):
        for metric in ('sr_pp', 'spl_pp'):
            values = [difference(full[comparison][seed], outside)[metric]
                      for seed in SEEDS]
            print(f'{comparison} outside-screen 1583 {metric}: '
                  f'mean {statistics.mean(values):+.6f}, '
                  f'sample SD {statistics.stdev(values):.6f}')


if __name__ == '__main__':
    main()
