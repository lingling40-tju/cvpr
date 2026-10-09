"""Independent arithmetic on compact training endings and prior SFT metrics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--paper-experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.directory / 'report.json').read_text())
    compact = a.directory / 'training_terminal_records.jsonl'
    assert sha(compact) == report['training_compact_sha256']
    assert report['model_calls'] == 0
    manifest = a.paper_experiment / 'fit512_manifest.json'
    assert sha(manifest) == report['fit_manifest_sha256']
    fit = set(json.loads(manifest.read_text())['episode_ids'])
    groups = defaultdict(list)
    for line in compact.open():
        row = json.loads(line); groups[row['run']].append(row)
    assert set(groups) == set(report['summaries']) and len(groups) == 2
    for label, rows in groups.items():
        arm = label.split('_')[2]
        original = a.paper_experiment / 'scale128' / (arm + '_seed11_rollout_budget.json')
        budget = json.loads(original.read_text())
        assert sha(original) == report['training_provenance'][label]['budget_report_sha256']
        assert budget['rollout_sha256'] == report['training_provenance'][label]['rollout_sha256']
        assert budget['training_config_sha256'] == report['training_provenance'][label]['config_sha256']
        assert len(rows) == 4096
        by_step = defaultdict(list)
        counts = Counter(); reasons = Counter(); budgets = Counter()
        for row in rows:
            by_step[row['step']].append(row)
            distance = row['distance_to_goal']
            near = distance is not None and distance <= 3
            assert row['logged_oracle_success'] == near
            assert row['distance_nonfinite'] == (distance is None)
            if distance is not None: assert math.isfinite(distance) and distance >= 0
            reward = row['total_reward']; assert math.isfinite(reward)
            assert math.isclose(reward, row['success_reward'] + row['ndtw_reward'], abs_tol=1e-8)
            timeout = row['end_reason'] in ('number of turns exceeded.', 'number of steps exceeded.')
            reasons[row['end_reason']] += 1
            budgets[str(row['turn_budget']) + '/' + str(row['step_budget'])] += 1
            counts['trajectories'] += 1
            counts['logged_task_successes'] += int(row['task_success'])
            counts['timeouts'] += int(timeout)
            counts['timeouts_inside_inclusive_three_m'] += int(timeout and near)
            counts['timeouts_inside_strict_three_m'] += int(timeout and distance is not None and distance < 3)
            counts['timeouts_exactly_three_m'] += int(timeout and distance == 3)
            counts['near_goal_timeouts_zero_reward'] += int(timeout and near and reward == 0)
            counts['near_goal_timeouts_task_failure'] += int(timeout and near and not row['task_success'])
            counts['terminal_generated_but_unexecuted_turns'] += int(row['terminal_extracted_actions'] > 0 and row['terminal_executed_actions'] == 0)
            counts['failed_stop_positive_ndtw'] += int(row['end_reason'] == 'stopped but goal not reached.' and row['ndtw_reward'] > 0)
            if timeout: assert reward == 0 and not row['task_success']
        assert sorted(by_step) == list(range(1, 129))
        for step, values in by_step.items():
            assert sorted(r['trajectory_index'] for r in values) == list(range(32))
            members = Counter(r['episode_id'] for r in values)
            assert set(members.values()) == {4}
            assert sorted(members) == budget['steps'][step - 1]['episode_ids']
        for lo, hi in ((1, 64), (65, 128)):
            members = Counter(r['episode_id'] for r in rows if lo <= r['step'] <= hi)
            assert set(members) == fit and set(members.values()) == {4}
        assert dict(counts) == report['summaries'][label]['counts']
        assert dict(reasons) == report['summaries'][label]['end_reasons']
        assert dict(budgets) == report['summaries'][label]['logged_turn_step_budget_histogram']
    sft = a.paper_experiment / 'completed20261009/posthoc_terminal_proposals/episodes.jsonl'
    assert sha(sft) == report['sft_diagnostic_sha256']
    sft_rows = [json.loads(line)['models']['positive_initial_sft'] for line in sft.open()]
    sft_counts = {
        'episodes': len(sft_rows), 'successes': sum(r['success'] for r in sft_rows),
        'forced_turn_limit_stops': sum(r['early_stop_reason'] == 'max_turns_reached' for r in sft_rows),
        'successful_forced_turn_limit_stops': sum(r['early_stop_reason'] == 'max_turns_reached' and r['success'] == 1 for r in sft_rows),
    }
    assert sft_counts == report['sft_observed_counts']
    output = {'schema': 'positive_budget_alignment_compact_independent_recount_v1', 'status': 'PASS',
              'training_records': 8192, 'training_runs': 2, 'sft_observed_counts': sft_counts,
              'report_sha256': sha(a.directory / 'report.json'), 'compact_sha256': sha(compact),
              'verifier_sha256': sha(Path(__file__)),
              'scope': 'Independent compact counts and original budget provenance; not second raw simulation, counterfactual reward replay or causal identification.'}
    with a.output.open('x') as stream: stream.write(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
