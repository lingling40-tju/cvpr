"""Post-hoc reward ordering within completed n=4 fit groups; no policy calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    assert sha(a.source) == '62d6576418ab324fac86cd8127ebf2b1cbd155340fd8b4b94497190fd460c790'
    assert sha(a.parent_report) == 'bf8b975b1d70fc3c1d4759e2a45005ac816d4e9d8f47a5dbe2d57796c275c727'
    groups = defaultdict(list)
    for line in a.source.open():
        x = json.loads(line)
        groups[(x['run'], x['step'], x['episode_id'])].append(x)
    summaries, cases = defaultdict(Counter), []
    for (run, step, eid), rows in groups.items():
        assert len(rows) == 4 and len({x['trajectory_index'] for x in rows}) == 4
        assert len({(x['turn_budget'], x['step_budget']) for x in rows}) == 1
        assert all(x['total_reward'] >= 0 for x in rows)
        near = [x for x in rows if x['end_reason'] in ('number of turns exceeded.', 'number of steps exceeded.') and x['distance_to_goal'] is not None and x['distance_to_goal'] < 3]
        farstop = [x for x in rows if x['end_reason'] == 'stopped but goal not reached.' and x['distance_to_goal'] is not None and x['distance_to_goal'] > 3 and x['ndtw_reward'] > 0]
        mean = sum(x['total_reward'] for x in rows) / 4
        c = summaries[run]
        c['groups'] += 1
        c['trajectories'] += 4
        c['near_timeout_trajectories'] += len(near)
        c['near_timeouts_below_group_mean'] += sum(x['total_reward'] < mean for x in near)
        c['near_timeouts_equal_group_mean'] += sum(x['total_reward'] == mean for x in near)
        c['groups_with_near_timeout'] += bool(near)
        c['groups_with_near_timeout_and_far_failed_stop'] += bool(near and farstop)
        pairs = [(x, y) for x in near for y in farstop if x['total_reward'] < y['total_reward'] and x['distance_to_goal'] < y['distance_to_goal']]
        c['far_failed_stop_over_near_timeout_pairs'] += len(pairs)
        c['groups_with_no_logged_success'] += not any(x['task_success'] for x in rows)
        c['no_success_groups_with_positive_failed_stop'] += not any(x['task_success'] for x in rows) and bool(farstop)
        c['nonfinite_distance_records'] += sum(x['distance_to_goal'] is None for x in rows)
        if pairs:
            cases.append({'run': run, 'step': step, 'episode_id': eid,
                'pairs': [{'near_trajectory_index': x['trajectory_index'], 'far_trajectory_index': y['trajectory_index'],
                    'near_distance': x['distance_to_goal'], 'far_distance': y['distance_to_goal'],
                    'near_reward': x['total_reward'], 'far_reward': y['total_reward'],
                    'far_ndtw_reward': y['ndtw_reward']} for x, y in pairs]})
    assert len(groups) == 2048 and len(summaries) == 2
    assert all(c['groups'] == 1024 and c['trajectories'] == 4096 for c in summaries.values())
    out = {'schema': 'posthoc_within_group_terminal_reward_order_v1', 'model_calls': 0,
        'source_compact_sha256': sha(a.source), 'parent_report_sha256': sha(a.parent_report),
        'exporter_sha256': sha(Path(__file__)), 'group_unit': 'run, optimizer step, episode ID; four recorded trajectories',
        'summaries': {run: dict(c) for run, c in summaries.items()}, 'rank_conflict_cases': cases,
        'scope': 'Descriptive reward/terminal-distance ordering on repeated fit rollouts. Mean-centered reward sign is not the positive-credit candidate advantage, semantic truth, evaluation SR, physical counterfactual or causal explanation. No frozen training, evaluation or gate is changed.'}
    with a.output.open('x') as f:
        f.write(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['summaries'], indent=2))


if __name__ == '__main__':
    main()
