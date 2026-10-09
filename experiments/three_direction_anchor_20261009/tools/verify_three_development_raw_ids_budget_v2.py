"""Final independent raw-ID/metric check after the frozen pilot evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


EVALUATION_SHA = 'bbf345f09bf9605e28873337752520c9c90523782983e8c5971dac956bfd0564'
MANIFEST_SHA = '8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    state = a.root / 'runlogs/development_suite'
    assert (state / 'suite.completed').exists() and not (state / 'suite.failed').exists()
    assert sha(a.root / 'runlogs/freeze/evaluation_identity.json') == EVALUATION_SHA
    manifest = a.root / 'prepared_data/development256.json'
    assert sha(manifest) == MANIFEST_SHA
    m = json.loads(manifest.read_text())
    ids = [str(i) for i in m['episode_ids']]
    assert len(ids) == len(set(ids)) == 256 and len(set(m['scene_ids'])) == 8
    labels = ['positive_initial_sft'] + ['td_' + v + '_n4_seed11' for v in ('grpo_anchor', 'turn_rloo', 'srgpo')]
    result = a.root / 'runlogs/development256'
    values = {}
    files = {}
    for label in labels:
        assert (result / (label + '.completed')).exists() and not (result / (label + '.failed')).exists()
        rows = {}
        for shard in range(4):
            log = result / label / ('shard_%02d' % shard) / 'log'
            selected = ids[shard::4]
            assert {f.name for f in log.glob('stats_*_0.json')} == {'stats_' + i + '_0.json' for i in selected}
            for eid in selected:
                f = log / ('stats_' + eid + '_0.json')
                v = json.loads(f.read_text())
                # The original evaluator stores its internal ID under "id".
                assert str(v['id']) == eid
                assert v.get('early_stop_reason') != 'inference_error'
                s, spl = v['success'], v['spl']
                assert s in (0, 1) and type(spl) in (int, float) and math.isfinite(spl) and 0 <= spl <= 1
                rows[eid] = [int(s), float(spl)]
                files[label + '/' + ('shard_%02d' % shard) + '/' + f.name] = sha(f)
        assert set(rows) == set(ids)
        values[label] = rows
    aggregate = json.loads((state / 'three_direction_report.json').read_text())
    comparisons = {}
    for method in ('grpo_anchor', 'turn_rloo', 'srgpo'):
        candidate = 'td_' + method + '_n4_seed11'
        for ref in ('sft', 'grpo'):
            if method == 'grpo_anchor' and ref == 'grpo':
                continue
            control = 'positive_initial_sft' if ref == 'sft' else 'td_grpo_anchor_n4_seed11'
            sr = 100 * sum(values[candidate][i][0] - values[control][i][0] for i in ids) / 256
            spl = 100 * sum(values[candidate][i][1] - values[control][i][1] for i in ids) / 256
            name = method + '_vs_' + ref
            report = aggregate['comparisons'][name]
            assert report['control'] == control and report['candidate'] == candidate
            assert math.isclose(sr, report['paired_sr_points'], abs_tol=1e-10, rel_tol=0)
            assert math.isclose(spl, report['paired_spl_points'], abs_tol=1e-10, rel_tol=0)
            comparisons[name] = {'paired_sr_points': sr, 'paired_spl_points': spl}
        vs_sft = comparisons[method + '_vs_sft']
        vs_grpo = comparisons.get(method + '_vs_grpo')
        eligible = all(v >= 2 for v in vs_sft.values()) and (vs_grpo is None or all(v >= 2 for v in vs_grpo.values()))
        assert eligible == aggregate['gates'][method]['eligible_for_scale']
    out = {'schema': 'three_direction_final_raw_id_independent_recount_v1', 'status': 'PASS',
           'raw_internal_ids_checked': 1024, 'episodes_per_model': 256, 'models': 4,
           'inference_errors': 0, 'comparisons': comparisons, 'raw_stats_sha256': files,
           'aggregate_sha256': sha(state / 'three_direction_report.json'),
           'verifier_sha256': sha(Path(__file__)),
           'scope': 'Additional raw internal-ID and paired-metric verification; Original training/inference/metric sources unchanged; transparent per-episode training-audit revision.'}
    with a.output.open('x') as f:
        f.write(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'raw_stats_sha256'}))


if __name__ == '__main__':
    main()
