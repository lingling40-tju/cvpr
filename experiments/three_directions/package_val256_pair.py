"""Export a checked, compact paired record for a completed 256-episode pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(root: Path, label: str, ids: list[str]) -> dict[str, dict]:
    assert (root / f'{label}.completed').exists(), label
    valid = json.loads((root / f'{label}.validated.json').read_text())
    assert valid['label'] == label and valid['episodes'] == len(ids)
    assert valid['inference_errors'] == 0
    rows = {}
    for shard in range(4):
        folder = root / label / f'shard_{shard:02d}'
        summary = json.loads((folder / 'summary.json').read_text())
        expected = ids[shard::4]
        assert len(expected) == summary['count'] == 64
        assert set(map(str, summary['episode_ids'])) == set(expected)
        assert summary['inference_errors'] == 0
        for eid in expected:
            row = json.loads((folder / 'log' / f'stats_{eid}_0.json').read_text())
            assert str(row['id']) == eid
            assert row.get('early_stop_reason') != 'inference_error'
            assert math.isfinite(float(row['spl'])) and 0 <= float(row['spl']) <= 1
            assert math.isfinite(float(row['distance_to_goal']))
            rows[eid] = row
    assert set(rows) == set(ids)
    return rows


def check_metrics(rows: dict[str, dict], ids: list[str], metrics: dict) -> None:
    assert metrics['count'] == len(ids) == 256 and metrics['inference_errors'] == 0
    assert metrics['successes'] == sum(bool(rows[eid]['success']) for eid in ids)
    for key, observed in (
        ('sr', sum(bool(rows[eid]['success']) for eid in ids) / 256),
        ('spl', sum(float(rows[eid]['spl']) for eid in ids) / 256),
        ('mean_distance_to_goal',
         sum(float(rows[eid]['distance_to_goal']) for eid in ids) / 256),
    ):
        assert math.isclose(metrics[key], observed, abs_tol=1e-12), key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ids = [str(x) for x in manifest['episode_ids']]
    scenes = [str(x) for x in manifest['scene_ids']]
    assert len(ids) == len(set(ids)) == len(scenes) == 256
    assert len(set(scenes)) == 11
    analysis = json.loads(args.analysis.read_text())
    assert analysis['split'] == 'val_unseen' and analysis['episodes'] == 256
    assert analysis['candidate'] == args.candidate and analysis['control'] == args.control
    assert analysis['manifest_sha256'] == digest(manifest_path)
    candidate = load_model(root, args.candidate, ids)
    control = load_model(root, args.control, ids)
    check_metrics(candidate, ids, analysis['candidate_metrics'])
    check_metrics(control, ids, analysis['control_metrics'])
    sr_pp = 100 * sum(bool(candidate[eid]['success']) - bool(control[eid]['success'])
                      for eid in ids) / 256
    spl_pp = 100 * sum(float(candidate[eid]['spl']) - float(control[eid]['spl'])
                       for eid in ids) / 256
    paired = analysis['paired']
    assert math.isclose(sr_pp, paired['sr_pp'], abs_tol=1e-12)
    assert math.isclose(spl_pp, paired['spl_pp'], abs_tol=1e-12)
    assert paired['candidate_only_successes'] == sum(
        bool(candidate[eid]['success']) and not bool(control[eid]['success'])
        for eid in ids)
    assert paired['control_only_successes'] == sum(
        bool(control[eid]['success']) and not bool(candidate[eid]['success'])
        for eid in ids)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    compact = output / 'paired_episodes.jsonl'
    with compact.open('w') as stream:
        for eid, scene in zip(ids, scenes):
            row = {'episode_id': eid, 'scene_id': scene}
            for name, model in (('candidate', candidate), ('control', control)):
                source = model[eid]
                row.update({
                    f'{name}_success': bool(source['success']),
                    f'{name}_spl': float(source['spl']),
                    f'{name}_distance_to_goal': float(source['distance_to_goal']),
                    f'{name}_oracle_success': bool(source['oracle_success']),
                    f'{name}_early_stop_reason': source.get('early_stop_reason'),
                })
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    analysis_copy = output / 'analysis.json'
    analysis_copy.write_bytes(args.analysis.read_bytes())
    package = {
        'candidate': args.candidate,
        'control': args.control,
        'episodes': 256,
        'manifest_sha256': digest(manifest_path),
        'analysis_sha256': digest(analysis_copy),
        'paired_episodes_sha256': digest(compact),
        'paired_sr_pp': sr_pp,
        'paired_spl_pp': spl_pp,
    }
    (output / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
