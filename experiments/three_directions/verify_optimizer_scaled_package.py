"""Independently recompute optimizer results from compact paired episode rows."""

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
DATASET_SHA = '2af6483b4b2f4229f5753d1cbca5f2214567effaa8baee91235310d2083411ea'
SEEDS = (11, 22, 33)


def close(left: float, right: float) -> None:
    assert math.isclose(left, right, rel_tol=0, abs_tol=1e-10), (left, right)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path, ids: list[str], scenes: list[str]) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(ids)
    assert [(str(x['episode_id']), str(x['scene_id'])) for x in rows] == list(zip(ids, scenes))
    for row in rows:
        for arm in ('candidate', 'control'):
            data = row[arm]
            assert type(data['success']) is bool
            assert math.isfinite(float(data['spl'])) and 0 <= float(data['spl']) <= 1
            assert math.isfinite(float(data['distance_to_goal_m']))
            assert data['early_stop_reason'] != 'inference_error'
    return {str(row['episode_id']): row for row in rows}


def metrics(rows: dict[str, dict], ids: list[str], arm: str) -> dict:
    data = [rows[eid][arm] for eid in ids]
    n = len(data)
    return {'count': n, 'successes': sum(x['success'] for x in data),
            'sr': sum(x['success'] for x in data) / n,
            'spl': sum(float(x['spl']) for x in data) / n,
            'mean_distance_to_goal': sum(float(x['distance_to_goal_m']) for x in data) / n}


def difference(rows: dict[str, dict], ids: list[str]) -> dict:
    values = [rows[eid] for eid in ids]
    n = len(values)
    return {
        'sr_pp': 100 * sum(x['candidate']['success'] - x['control']['success'] for x in values) / n,
        'spl_pp': 100 * sum(float(x['candidate']['spl']) - float(x['control']['spl']) for x in values) / n,
        'candidate_only_successes': sum(x['candidate']['success'] and
                                        not x['control']['success'] for x in values),
        'control_only_successes': sum(x['control']['success'] and
                                      not x['candidate']['success'] for x in values),
    }


def check_result(root: Path, mode: str, count: int) -> tuple[dict[int, dict], list[str], list[str]]:
    prefix = 'val256' if count == 256 else 'full1839'
    manifest_path = root / f'{prefix}_manifest.json'
    assert digest(manifest_path) == MANIFEST_SHA[count]
    manifest = json.loads(manifest_path.read_text())
    ids = [str(x) for x in manifest['episode_ids']]
    scenes = [str(x) for x in manifest['scene_ids']]
    assert len(ids) == len(set(ids)) == len(scenes) == count
    analysis = json.loads((root / f'{prefix}_analysis.json').read_text())
    assert analysis['mode'] == mode and analysis['split'] == 'val_unseen'
    assert analysis['episodes_per_seed'] == count
    assert analysis['manifest_sha256'] == MANIFEST_SHA[count]
    assert analysis['seeds'] == list(SEEDS) and len(analysis['pairs']) == 3
    all_rows = {}
    for seed, pair in zip(SEEDS, analysis['pairs']):
        assert pair['seed'] == seed
        assert pair['candidate'] == f'{mode}_128_seed{seed}'
        assert pair['control'] == f'branch_control128_seed{seed}'
        rows = read_rows(root / f'paired_{count}_seed{seed}.jsonl', ids, scenes)
        all_rows[seed] = rows
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
    for key in ('sr_pp', 'spl_pp'):
        values = [difference(all_rows[seed], ids)[key] for seed in SEEDS]
        summary = analysis['summary'][key]
        for observed, expected in zip(values, summary['values']):
            close(observed, expected)
        close(statistics.mean(values), summary['mean'])
        close(statistics.stdev(values), summary['sample_sd'])
    print(f'{mode} {count}: paired SR {analysis["summary"]["sr_pp"]["mean"]:+.6f} pp, '
          f'SPL {analysis["summary"]["spl_pp"]["mean"]:+.6f} pp')
    return all_rows, ids, scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('package_dir', type=Path)
    parser.add_argument('--mode', choices=('group4', 'kl_anchor', 'dynamic'), required=True)
    args = parser.parse_args()
    root = args.package_dir
    filenames = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert filename not in filenames
        filenames.add(filename)
        assert digest(root / filename) == expected, filename
    required = {f'paired_{count}_seed{seed}.jsonl' for count in (256, 1839)
                for seed in SEEDS}
    required |= {'val256_manifest.json', 'full1839_manifest.json',
                 'val256_analysis.json', 'full1839_analysis.json'}
    required |= {f'seed{seed}_train_audit.json' for seed in SEEDS}
    assert filenames == required
    for seed in SEEDS:
        audit = json.loads((root / f'seed{seed}_train_audit.json').read_text())
        assert audit['mode'] == args.mode and audit['seed'] == seed
        assert audit['steps'] == 128 and audit['unique_train_episodes'] == 512
        assert audit['matched_train_episode_sets_at_each_step'] == 128
        assert audit['dataset_sha256'] == DATASET_SHA
        if args.mode == 'dynamic':
            assert audit['candidate_final_rollouts'] == 1024
            assert audit['control_rollouts'] == 1024
            assert audit['additional_simulator_rollouts'] > 0
            assert audit['total_candidate_simulator_rollouts'] == (
                1024 + audit['additional_simulator_rollouts'])
        else:
            assert audit['candidate_rollouts'] == 128 * 4 * (4 if args.mode == 'group4' else 2)
        if args.mode == 'kl_anchor':
            assert audit['actor_kl_coef'] == 0.001
            assert len(audit['actor_kl_losses_tensorboard']) == 128
    screen, screen_ids, screen_scenes = check_result(root, args.mode, 256)
    full, full_ids, full_scenes = check_result(root, args.mode, 1839)
    full_scene = dict(zip(full_ids, full_scenes))
    assert set(screen_ids) < set(full_ids)
    assert all(full_scene[eid] == scene for eid, scene in zip(screen_ids, screen_scenes))
    disjoint_ids = [eid for eid in full_ids if eid not in set(screen_ids)]
    assert len(disjoint_ids) == 1583
    for key in ('sr_pp', 'spl_pp'):
        values = [difference(full[seed], disjoint_ids)[key] for seed in SEEDS]
        print(f'{args.mode} screen-disjoint 1583 {key}: mean {statistics.mean(values):+.6f}, '
              f'sample SD {statistics.stdev(values):.6f}')


if __name__ == '__main__':
    main()
