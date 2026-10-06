"""Independent raw per-episode recount for post-result n=8 sensitivity."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

root, candidate, control_report, n4_report = sys.argv[1:]
root = Path(root)
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['split'] == 'val_seen'
ids = [str(x) for x in manifest['episode_ids']]
assert len(ids) == len(set(ids)) == 256
assert len(set(map(str, manifest['scene_ids']))) == 38
assert (root / f'{candidate}.completed').is_file()


def read(label):
    assert (root / f'{label}.completed').is_file()
    rows = {}
    for shard in range(4):
        folder = root / label / f'shard_{shard:02d}' / 'log'
        actual = list(folder.glob('stats_*_0.json'))
        expected = ids[shard::4]
        assert len(actual) == len(expected) == 64, (label, shard, len(actual))
        for episode_id in expected:
            record = json.loads((folder / f'stats_{episode_id}_0.json').read_text())
            assert str(record.get('episode_id', episode_id)) == episode_id
            assert record.get('early_stop_reason') != 'inference_error'
            assert episode_id not in rows
            success = record['success']
            if isinstance(success, bool):
                success = int(success)
            assert isinstance(success, (int, float)) and math.isfinite(success)
            assert success in (0, 1)
            spl = record['spl']
            assert not isinstance(spl, bool) and isinstance(spl, (int, float))
            assert math.isfinite(spl) and 0 <= spl <= 1
            rows[episode_id] = record
    assert set(rows) == set(ids)
    return rows

labels = ['qwen3_exact_control_64step_seed11', 'norm_terminal_rloo_64step_seed11', candidate]
data = {label: read(label) for label in labels}
reports = [json.loads(Path(control_report).read_text()), json.loads(Path(n4_report).read_text())]
for baseline, report in zip(labels[:2], reports):
    assert report['control'] == baseline and report['candidate'] == candidate
    a, b = data[baseline], data[candidate]
    n = len(ids)
    a_sr = sum(bool(a[i]['success']) for i in ids) / n
    b_sr = sum(bool(b[i]['success']) for i in ids) / n
    a_spl = sum(float(a[i]['spl']) for i in ids) / n
    b_spl = sum(float(b[i]['spl']) for i in ids) / n
    assert report['episodes'] == n and report['scenes'] == 38
    assert report['control_successes'] == round(a_sr * n)
    assert report['candidate_successes'] == round(b_sr * n)
    assert math.isclose(report['paired_sr_points'], 100 * (b_sr - a_sr), abs_tol=1e-10)
    assert math.isclose(report['paired_spl_points'], 100 * (b_spl - a_spl), abs_tol=1e-10)
    print(json.dumps({'baseline': baseline, 'candidate': candidate,
                      'episodes': n, 'scenes': 38, 'inference_errors': 0,
                      'baseline_successes': round(a_sr * n),
                      'candidate_successes': round(b_sr * n),
                      'paired_sr_points': 100 * (b_sr - a_sr),
                      'paired_spl_points': 100 * (b_spl - a_spl)}))
