"""Independent CPU recount of raw episode IDs and paired development metrics."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = '8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = ROOT / 'prepared_data/development256.json'
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('development manifest changed')
    manifest = json.loads(manifest_path.read_text())
    ids = [str(value) for value in manifest['episode_ids']]
    if len(ids) != 256 or len(set(ids)) != 256 or len(set(manifest['scene_ids'])) != 8:
        raise ValueError('fixed episode/scene membership differs')
    result = ROOT / 'runlogs/development256'
    labels = ['positive_initial_sft', 'td_turn_rloo_n4_seed11',
              'td_outcome_consistent_rloo_n4_seed11']
    all_values = {}
    raw_hashes = {}
    for label in labels:
        if not (result / (label + '.completed')).exists() or (result / (label + '.failed')).exists():
            raise ValueError('model completion marker missing or failed: ' + label)
        values = {}
        for shard in range(4):
            log = result / label / f'shard_{shard:02d}' / 'log'
            wanted = ids[shard::4]
            found = {path.name for path in log.glob('stats_*_0.json')}
            if found != {'stats_' + eid + '_0.json' for eid in wanted}:
                raise ValueError('raw per-shard filenames differ: ' + label)
            for eid in wanted:
                path = log / ('stats_' + eid + '_0.json')
                row = json.loads(path.read_text())
                if str(row.get('id')) != eid or row.get('early_stop_reason') == 'inference_error':
                    raise ValueError('wrong internal ID or inference error: ' + label + '/' + eid)
                success, spl = row.get('success'), row.get('spl')
                if success not in (0, 1) or not isinstance(spl, (int, float)) or \
                        not math.isfinite(float(spl)) or not 0 <= float(spl) <= 1:
                    raise ValueError('invalid raw SR/SPL: ' + label + '/' + eid)
                values[eid] = (int(success), float(spl))
                raw_hashes[label + '/shard_' + str(shard).zfill(2) + '/' + path.name] = sha(path)
        if set(values) != set(ids):
            raise ValueError('raw episode coverage differs: ' + label)
        all_values[label] = values

    state = ROOT / 'runlogs/development_suite'
    report = json.loads((state / 'outcome_report.json').read_text())
    expected_pairs = {
        'outcome_vs_sft': ('td_outcome_consistent_rloo_n4_seed11', 'positive_initial_sft'),
        'outcome_vs_previous_rloo': ('td_outcome_consistent_rloo_n4_seed11', 'td_turn_rloo_n4_seed11'),
    }
    comparison_recount = {}
    for name, (candidate, control) in expected_pairs.items():
        sr = 100 * sum(all_values[candidate][eid][0] - all_values[control][eid][0] for eid in ids) / 256
        spl = 100 * sum(all_values[candidate][eid][1] - all_values[control][eid][1] for eid in ids) / 256
        declared = report['comparisons'][name]
        if not math.isclose(sr, declared['paired_sr_points'], abs_tol=1e-10, rel_tol=0) or \
                not math.isclose(spl, declared['paired_spl_points'], abs_tol=1e-10, rel_tol=0):
            raise ValueError('raw paired metrics disagree: ' + name)
        analysis = json.loads((state / (name + '.json')).read_text())
        independent = json.loads((state / (name + '.independent.json')).read_text())
        for item in (analysis, independent):
            if not math.isclose(sr, item['paired_sr_points'], abs_tol=1e-10, rel_tol=0) or \
                    not math.isclose(spl, item['paired_spl_points'], abs_tol=1e-10, rel_tol=0):
                raise ValueError('raw metrics disagree with analysis/compact check: ' + name)
        comparison_recount[name] = {'paired_sr_points': sr, 'paired_spl_points': spl}
    eligible = all(report['gate_checks'].values())
    if bool(report['eligible_for_scale']) != eligible:
        raise ValueError('scale eligibility differs from gate booleans')
    output = {
        'schema': 'outcome_consistent_rloo_final_raw_id_recount_v1',
        'status': 'PASS',
        'identity_sha256': report['identity_sha256'],
        'episodes_per_model': 256,
        'scenes': 8,
        'models': 3,
        'raw_internal_ids_checked': 768,
        'inference_errors': 0,
        'comparisons': comparison_recount,
        'eligible_for_scale': eligible,
        'raw_stats_sha256': raw_hashes,
        'outcome_report_sha256': sha(state / 'outcome_report.json'),
        'verifier_sha256': sha(Path(__file__)),
        'scope': 'Independent exact-ID, SR/SPL and frozen-gate recount from raw Habitat stats JSON.'
    }
    path = state / 'final_raw_id_independent_recount.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ('status', 'raw_internal_ids_checked', 'comparisons', 'eligible_for_scale')}))


if __name__ == '__main__':
    main()
