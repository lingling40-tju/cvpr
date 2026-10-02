"""Recompute a compact fixed-256 pilot package without raw simulator logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('package_dir', type=Path)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parent / 'val256/manifest.json')
    args = parser.parse_args()
    package_dir = args.package_dir
    compact = package_dir / 'paired_episodes.jsonl'
    analysis_file = package_dir / 'analysis.json'
    package = json.loads((package_dir / 'package.json').read_text())
    analysis = json.loads(analysis_file.read_text())
    manifest = json.loads(args.manifest.read_text())
    rows = [json.loads(line) for line in compact.read_text().splitlines()]
    ids = [str(x) for x in manifest['episode_ids']]
    scenes = [str(x) for x in manifest['scene_ids']]
    assert len(ids) == len(set(ids)) == len(scenes) == len(rows) == 256
    assert [(r['episode_id'], r['scene_id']) for r in rows] == list(zip(ids, scenes))
    assert len(set(scenes)) == analysis['scenes'] == 11
    assert package['episodes'] == analysis['episodes'] == 256
    assert package['manifest_sha256'] == analysis['manifest_sha256'] == sha256(args.manifest)
    assert package['analysis_sha256'] == sha256(analysis_file)
    assert package['paired_episodes_sha256'] == sha256(compact)
    assert package['candidate'] == analysis['candidate']
    assert package['control'] == analysis['control']

    for prefix, field in (('candidate', 'candidate_metrics'), ('control', 'control_metrics')):
        metrics = analysis[field]
        assert metrics['count'] == 256 and metrics['inference_errors'] == 0
        successes = sum(bool(r[f'{prefix}_success']) for r in rows)
        assert successes == metrics['successes']
        for key, column in (('sr', f'{prefix}_success'),
                            ('spl', f'{prefix}_spl'),
                            ('mean_distance_to_goal', f'{prefix}_distance_to_goal')):
            values = [float(r[column]) for r in rows]
            assert all(math.isfinite(x) for x in values)
            assert math.isclose(sum(values) / 256, metrics[key], abs_tol=1e-12)
    sr_pp = 100 * sum(int(r['candidate_success']) - int(r['control_success'])
                      for r in rows) / 256
    spl_pp = 100 * sum(r['candidate_spl'] - r['control_spl'] for r in rows) / 256
    paired = analysis['paired']
    for value, expected in ((sr_pp, paired['sr_pp']),
                            (spl_pp, paired['spl_pp']),
                            (sr_pp, package['paired_sr_pp']),
                            (spl_pp, package['paired_spl_pp'])):
        assert math.isclose(value, expected, abs_tol=1e-12)
    assert paired['candidate_only_successes'] == sum(
        r['candidate_success'] and not r['control_success'] for r in rows)
    assert paired['control_only_successes'] == sum(
        r['control_success'] and not r['candidate_success'] for r in rows)
    print(f"verified {package['candidate']} vs {package['control']}: "
          f"256 episodes; SR {sr_pp:+.6f} pp, SPL {spl_pp:+.6f} pp")


if __name__ == '__main__':
    main()
