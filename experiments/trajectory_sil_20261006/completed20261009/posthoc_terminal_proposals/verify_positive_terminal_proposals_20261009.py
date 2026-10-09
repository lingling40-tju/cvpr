"""Independent compact arithmetic recount; does not infer physical execution."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--original-compact', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.diagnostic / 'report.json').read_text())
    data = a.diagnostic / 'episodes.jsonl'
    assert sha(data) == report['diagnostic_compact_sha256']
    assert sha(a.original_compact) == report['source_compact_sha256']
    assert sha(a.manifest) == report['manifest_sha256']
    m = json.loads(a.manifest.read_text())
    rows = [json.loads(l) for l in data.read_text().splitlines()]
    originals = [json.loads(l) for l in a.original_compact.read_text().splitlines()]
    assert len(rows) == 1839 and len({r['episode_id'] for r in rows}) == 1839
    assert [r['episode_id'] for r in rows] == [str(v) for v in m['episode_ids']]
    assert [r['scene_id'] for r in rows] == m['scene_ids']
    labels = list(report['summaries'])
    assert len(labels) == 7 and report['model_calls'] == 0 and report['inference_errors'] == 0
    for row, source in zip(rows, originals):
        assert row['episode_id'] == source['episode_id']
        assert set(row['models']) == set(labels) == set(source['models'])
        for label in labels:
            v = row['models'][label]
            for key in ('success', 'path_length', 'distance_to_goal', 'early_stop_reason'):
                assert v[key] == source['models'][label][key]
            assert len(v['parsed_action_ids_by_turn']) == v['saved_assistant_turns']
            assert 1 <= v['saved_assistant_turns'] <= 12
            assert all(t and all(x is None or type(x) is int and x in (0, 1, 2, 3) for x in t)
                       for t in v['parsed_action_ids_by_turn'])
    for label in labels:
        vals = [r['models'][label] for r in rows]
        bad = [v for v in vals if not v['success']]
        expected = {'episodes': len(vals), 'successes': sum(v['success'] for v in vals),
                    'termination_reason_counts': dict(Counter('unclassified' if v['early_stop_reason'] is None
                                                      else v['early_stop_reason'] for v in vals)),
                    'saved_assistant_turn_histogram': {str(k): n for k, n in Counter(v['saved_assistant_turns'] for v in vals).items()},
                    'median_saved_assistant_turns': statistics.median(v['saved_assistant_turns'] for v in vals),
                    'mean_saved_assistant_turns': statistics.mean(v['saved_assistant_turns'] for v in vals),
                    'last_turn_any_parsed_stop_proposal': sum(0 in v['parsed_action_ids_by_turn'][-1] for v in vals),
                    'last_turn_first_parsed_action_stop_proposal': sum(v['parsed_action_ids_by_turn'][-1][0] == 0 for v in vals),
                    'failure_last_turn_first_stop_proposal': sum(v['parsed_action_ids_by_turn'][-1][0] == 0 for v in bad),
                    'failure_median_path_m': statistics.median(v['path_length'] for v in bad),
                    'failure_median_terminal_distance_m': statistics.median(v['distance_to_goal'] for v in bad),
                    'failure_path_at_most_one_m': sum(v['path_length'] <= 1 for v in bad),
                    'failure_first_stop_proposal_by_three_saved_turns': sum(v['saved_assistant_turns'] <= 3
                        and v['parsed_action_ids_by_turn'][-1][0] == 0 for v in bad)}
        assert expected == report['summaries'][label], label
        if label != 'positive_initial_sft':
            lost = [r for r in rows if r['models']['positive_initial_sft']['success'] and not r['models'][label]['success']]
            expected_pair = {'sft_success_trained_failure': len(lost),
                             'trained_last_turn_first_stop_proposal': sum(r['models'][label]['parsed_action_ids_by_turn'][-1][0] == 0 for r in lost),
                             'trained_median_saved_turns': statistics.median(r['models'][label]['saved_assistant_turns'] for r in lost),
                             'sft_median_saved_turns': statistics.median(r['models']['positive_initial_sft']['saved_assistant_turns'] for r in lost)}
            assert expected_pair == report['paired_sft_lost_successes'][label], label
    out = {'schema': 'positive_terminal_proposal_compact_independent_recount_v1',
           'status': 'PASS', 'episodes': 1839, 'models': 7, 'summaries_and_pairing_agree': True,
           'diagnostic_compact_sha256': sha(data), 'source_compact_sha256': sha(a.original_compact),
           'scope': 'Independent arithmetic on exported proposal IDs and existing metrics; no raw-text reparse, physical-action trace, or causal verification.'}
    with a.output.open('x') as f:
        f.write(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
