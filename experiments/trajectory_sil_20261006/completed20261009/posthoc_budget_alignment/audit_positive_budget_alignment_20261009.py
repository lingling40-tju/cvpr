"""CPU post-hoc audit of logged timeout outcomes; never runs a policy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

SOURCE_SHAS = {
    'vlnce_server/env.py': '39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a',
    'vlnce_server/semantic_reward/env.py': '6d90aed3c919cf76d892aa7984b6c5586c04e07483a26310584579aa13eb1ead',
    'eval/vlnce/eval_vlnce.py': '7dec491e633f09d7ba028a9e26b6503af1d94b3d27d007081bc1ad213192b328',
}
RAW_SHAS = {
    'control': 'c4b2e4bd3c940bd17f713044ae4ba497b10ab1bde6f0a527523138b9af408909',
    'candidate': '8a098ad4bd1486bb5c59c191010cf4ce8e270cbc5ca10b9072ab316f57ff67ad',
}
TIMEOUTS = ('number of turns exceeded.', 'number of steps exceeded.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--current-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    assert not a.output_dir.exists()
    identities = {}
    excerpts = {}
    for rel, expected in SOURCE_SHAS.items():
        old, current = a.root / rel, a.current_root / rel
        assert sha(old) == sha(current) == expected
        identities[rel] = {'old_sha256': expected, 'current_sha256': expected}
        lines = old.read_text().splitlines()
        ranges = {'vlnce_server/env.py': [(367, 452), (614, 621)],
                  'vlnce_server/semantic_reward/env.py': [(99, 108)],
                  'eval/vlnce/eval_vlnce.py': [(93, 116), (327, 340)]}[rel]
        excerpts[rel] = [{'line': i + 1, 'text': lines[i]} for lo, hi in ranges for i in range(lo - 1, hi)]
    manifest = a.root / 'prepared_data/fit512_manifest.json'
    assert sha(manifest) == '98dd7ec7bcf707dcfe62162761b26a3ff6f1170101bdbccf1c19e0ec0d18bc20'
    fit = set(json.loads(manifest.read_text())['episode_ids'])
    assert len(fit) == 512
    rows, summaries, provenance = [], {}, {}
    for arm, expected_raw in RAW_SHAS.items():
        label = 'positive_trajectory_' + arm + '_128step_seed11'
        run = a.root / 'runlogs' / label
        assert (run / 'completed').exists() and not (run / 'failed').exists()
        raw = a.root / 'verl_checkpoints' / label / 'rollout.jsonl'
        assert sha(raw) == expected_raw
        budget = a.root / 'runlogs/positive_scale' / (arm + '_seed11_rollout_budget.json')
        budget_report = json.loads(budget.read_text())
        assert budget_report['rollout_sha256'] == expected_raw
        assert budget_report['recorded_rollout_trajectories'] == 4096
        count = Counter(); reasons = Counter(); budgets = Counter(); epoch_members = [Counter(), Counter()]
        with raw.open() as stream:
            for step, line in enumerate(stream, 1):
                v = json.loads(line); assert v['step'] == step and len(v['info']) == 32
                members = Counter(str(t['episode_id']) for t in v['info'])
                assert len(members) == 8 and set(members.values()) == {4} and set(members) <= fit
                epoch_members[(step - 1) // 64].update(members)
                for index, t in enumerate(v['info']):
                    distance = float(t['distance_to_goal'])
                    assert not math.isnan(distance) and distance >= 0
                    near = distance <= 3.0
                    assert bool(t['orcale_success']) == near and t['done']
                    reward = float(t['total_reward']); assert math.isfinite(reward)
                    components = t['reward_components']
                    assert components['semantic_reward'] == components['success_floor'] == 0
                    assert math.isclose(reward, sum(components.values()), abs_tol=1e-8)
                    reason = t['end_reason']; timeout = reason in TIMEOUTS
                    success = bool(t['task_success'])
                    generated = t['gen_traj']; assert generated and len(generated) <= 13
                    last_extracted = len(generated[-1]['extracted_actions'])
                    last_executed = len(generated[-1]['executed_actions'])
                    row = {'run': label, 'step': step, 'trajectory_index': index,
                           'episode_id': str(t['episode_id']), 'end_reason': reason,
                           'distance_to_goal': distance if math.isfinite(distance) else None,
                           'distance_nonfinite': not math.isfinite(distance), 'logged_oracle_success': near,
                           'task_success': success, 'total_reward': reward,
                           'success_reward': float(components['success_reward']),
                           'ndtw_reward': float(components['ndtw_reward']),
                           'turn_budget': t['turn_budget'], 'step_budget': t['step_budget'],
                           'saved_generated_turns': len(generated),
                           'terminal_extracted_actions': last_extracted,
                           'terminal_executed_actions': last_executed}
                    assert 0 < row['turn_budget'] <= 12 and 0 < row['step_budget'] <= 36
                    budgets[str(row['turn_budget']) + '/' + str(row['step_budget'])] += 1
                    rows.append(row); count['trajectories'] += 1; reasons[reason] += 1
                    count['logged_task_successes'] += int(success)
                    count['timeouts'] += int(timeout)
                    count['timeouts_inside_inclusive_three_m'] += int(timeout and near)
                    count['timeouts_inside_strict_three_m'] += int(timeout and distance < 3)
                    count['timeouts_exactly_three_m'] += int(timeout and distance == 3)
                    count['near_goal_timeouts_zero_reward'] += int(timeout and near and reward == 0)
                    count['near_goal_timeouts_task_failure'] += int(timeout and near and not success)
                    count['terminal_generated_but_unexecuted_turns'] += int(last_extracted > 0 and last_executed == 0)
                    count['failed_stop_positive_ndtw'] += int(reason == 'stopped but goal not reached.' and components['ndtw_reward'] > 0)
                    if timeout:
                        assert not success and reward == 0 and components['ndtw_reward'] == 0
            assert step == 128
        assert all(set(c) == fit and set(c.values()) == {4} for c in epoch_members)
        assert count['trajectories'] == 4096
        summaries[label] = {'counts': dict(count), 'end_reasons': dict(reasons),
                            'logged_turn_step_budget_histogram': dict(budgets)}
        provenance[label] = {'rollout_sha256': expected_raw, 'budget_report_sha256': sha(budget),
                             'config_sha256': sha(run / 'config.txt')}
    diagnostic = a.root / 'runlogs/positive_posthoc_terminal20261009/episodes.jsonl'
    assert sha(diagnostic) == 'a8f6aaf6d467684ef6219448885a00e7c3f3509b196d40480c21001dfbbbee12'
    sft_counts = Counter()
    for line in diagnostic.open():
        t = json.loads(line)['models']['positive_initial_sft']
        sft_counts['episodes'] += 1; sft_counts['successes'] += int(t['success'])
        sft_counts['forced_turn_limit_stops'] += int(t['early_stop_reason'] == 'max_turns_reached')
        sft_counts['successful_forced_turn_limit_stops'] += int(t['early_stop_reason'] == 'max_turns_reached' and t['success'] == 1)
    assert sft_counts['episodes'] == 1839 and sft_counts['successes'] == 555
    a.output_dir.mkdir()
    compact = a.output_dir / 'training_terminal_records.jsonl'
    with compact.open('x') as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    report = {'schema': 'positive_posthoc_budget_alignment_audit_v1', 'model_calls': 0,
              'summaries': summaries, 'source_identities': identities, 'source_excerpts': excerpts,
              'training_provenance': provenance, 'fit_manifest_sha256': sha(manifest),
              'training_compact_sha256': sha(compact), 'sft_diagnostic_sha256': sha(diagnostic),
              'sft_observed_counts': dict(sft_counts), 'exporter_sha256': sha(Path(__file__)),
              'scope': 'Source and logged training-outcome audit, plus prior observed SFT evaluation counts; no policy calls, causal identification, counterfactual replay or new navigation benefit.'}
    with (a.output_dir / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('summaries', 'sft_observed_counts', 'scope')}))


if __name__ == '__main__':
    main()
