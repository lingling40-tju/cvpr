"""Apply the frozen advancement gate to paired candidate comparisons."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runlogs/development_suite'
RESULT = ROOT / 'runlogs/development256'


def main():
    identity = json.loads((ROOT / 'runlogs/freeze/identity.json').read_text())
    pairs = {}
    for name in ('outcome_vs_sft', 'outcome_vs_previous_rloo'):
        pair = json.loads((STATE / (name + '.json')).read_text())
        independent = json.loads((STATE / (name + '.independent.json')).read_text())
        if pair['paired_sr_points'] != independent['paired_sr_points'] or \
                pair['paired_spl_points'] != independent['paired_spl_points']:
            raise ValueError('compact verifier disagrees for ' + name)
        pairs[name] = pair
    candidate = json.loads((RESULT / 'td_outcome_consistent_rloo_n4_seed11.validated.json').read_text())
    sft = json.loads((RESULT / 'positive_initial_sft.validated.json').read_text())
    old = json.loads((RESULT / 'td_turn_rloo_n4_seed11.validated.json').read_text())
    if any(row['episodes'] != 256 or row['inference_errors'] != 0 for row in (candidate, sft, old)):
        raise ValueError('one arm lacks exact successful 256-episode coverage')
    gate = identity['gate']
    checks = {
        'vs_sft_sr': pairs['outcome_vs_sft']['paired_sr_points'] >= gate['vs_sft_sr_points'],
        'vs_sft_spl': pairs['outcome_vs_sft']['paired_spl_points'] >= gate['vs_sft_spl_points'],
        'vs_previous_rloo_sr': pairs['outcome_vs_previous_rloo']['paired_sr_points'] >= gate['vs_previous_rloo_sr_points'],
        'vs_previous_rloo_spl': pairs['outcome_vs_previous_rloo']['paired_spl_points'] >= gate['vs_previous_rloo_spl_points'],
    }
    report = {
        'schema': 'outcome_consistent_rloo_development_report_v1',
        'identity_sha256': identity['identity_sha256'],
        'manifest_sha256': identity['evaluation']['manifest_sha256'],
        'episodes': 256,
        'scenes': 8,
        'candidate': {'label': candidate['label'], 'successes': candidate['successes'],
                      'sr': 100 * candidate['successes'] / 256, 'spl': 100 * candidate['spl'],
                      'inference_errors': candidate['inference_errors']},
        'references': {
            'sft': {'successes': sft['successes'], 'sr': 100 * sft['successes'] / 256,
                    'spl': 100 * sft['spl']},
            'previous_turn_rloo': {'successes': old['successes'], 'sr': 100 * old['successes'] / 256,
                                   'spl': 100 * old['spl']},
        },
        'comparisons': {name: {
            'paired_sr_points': pair['paired_sr_points'],
            'paired_spl_points': pair['paired_spl_points'],
            'candidate_only_success': pair['candidate_only_success'],
            'control_only_success': pair['control_only_success'],
            'scene_bootstrap_95pct_sr_points': pair['scene_bootstrap_95pct_sr_points'],
            'scene_bootstrap_95pct_spl_points': pair['scene_bootstrap_95pct_spl_points'],
            'compact_sha256': pair['compact_sha256'],
            'independent_sha256': __import__('hashlib').sha256((STATE / (name + '.independent.json')).read_bytes()).hexdigest(),
        } for name, pair in pairs.items()},
        'gate_thresholds_pp': gate,
        'gate_checks': checks,
        'eligible_for_scale': all(checks.values()),
        'limitations': identity['limits'],
        'raw_id_independent_recount_required': True,
    }
    out = STATE / 'outcome_report.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'candidate_sr': report['candidate']['sr'], 'candidate_spl': report['candidate']['spl'],
                      'comparisons': {k: {m: v[m] for m in ('paired_sr_points', 'paired_spl_points')}
                                      for k, v in report['comparisons'].items()},
                      'gate_checks': checks, 'eligible_for_scale': report['eligible_for_scale']}))


if __name__ == '__main__':
    main()
