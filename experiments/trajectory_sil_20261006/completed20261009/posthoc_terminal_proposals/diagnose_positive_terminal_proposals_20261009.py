"""Post-hoc CPU description of completed outputs, not an action or causal audit."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


MANIFEST_SHA = '262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e'
COMPACT_SHA = '1435334c9c22153cbec61cc1478aaf123973c74905a2fd1914f2b870000107e0'
PARSER_SHA = '7dec491e633f09d7ba028a9e26b6503af1d94b3d27d007081bc1ad213192b328'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parser(source):
    if sha(source) != PARSER_SHA:
        raise ValueError('original evaluator parser changed')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActiveVlnAgent')
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('extract_result', 'extract_multi_result')]
    if len(functions) != 2:
        raise ValueError('parser methods missing')
    env = {'re': re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), env)
    target = type('Parser', (), {'forward_distance': 25, 'turn_angle': 15,
                                'extract_result': env['extract_result'],
                                'extract_multi_result': env['extract_multi_result']})()
    return target


def description(rows):
    failures = [r for r in rows if r['success'] == 0]
    return {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
            'termination_reason_counts': dict(Counter('unclassified' if r['early_stop_reason'] is None
                                                      else r['early_stop_reason'] for r in rows)),
            'saved_assistant_turn_histogram': dict(sorted(Counter(r['saved_assistant_turns'] for r in rows).items())),
            'median_saved_assistant_turns': statistics.median(r['saved_assistant_turns'] for r in rows),
            'mean_saved_assistant_turns': statistics.mean(r['saved_assistant_turns'] for r in rows),
            'last_turn_any_parsed_stop_proposal': sum(0 in r['parsed_action_ids_by_turn'][-1] for r in rows),
            'last_turn_first_parsed_action_stop_proposal': sum(r['parsed_action_ids_by_turn'][-1][0] == 0 for r in rows),
            'failure_last_turn_first_stop_proposal': sum(r['parsed_action_ids_by_turn'][-1][0] == 0 for r in failures),
            'failure_median_path_m': statistics.median(r['path_length'] for r in failures),
            'failure_median_terminal_distance_m': statistics.median(r['distance_to_goal'] for r in failures),
            'failure_path_at_most_one_m': sum(r['path_length'] <= 1 for r in failures),
            'failure_first_stop_proposal_by_three_saved_turns': sum(r['saved_assistant_turns'] <= 3
                and r['parsed_action_ids_by_turn'][-1][0] == 0 for r in failures)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    for suite in ('positive_scale', 'positive_extra', 'positive_matched_precision'):
        folder = a.root / 'runlogs' / suite
        if not (folder / 'suite.completed').exists() or (folder / 'suite.failed').exists():
            raise ValueError('original evaluation has not completed')
    manifest_path = a.root / 'prepared_data/positive_full1839.json'
    compact_path = a.root / 'runlogs/positive_matched_precision/val_unseen_with_sft.jsonl'
    if sha(manifest_path) != MANIFEST_SHA or sha(compact_path) != COMPACT_SHA:
        raise ValueError('frozen completed source differs')
    m = json.loads(manifest_path.read_text())
    ids = [str(i) for i in m['episode_ids']]
    source_rows = [json.loads(l) for l in compact_path.read_text().splitlines()]
    if len(ids) != 1839 or len(set(ids)) != 1839 or [r['episode_id'] for r in source_rows] != ids:
        raise ValueError('episode coverage differs')
    action_parser = parser(a.root / 'eval/vlnce/eval_vlnce.py')
    labels = ['positive_initial_sft'] + ['positive_trajectory_' + arm + '_128step_seed' + str(seed)
              for seed in (11, 22, 33) for arm in ('control', 'candidate')]
    records = [{'episode_id': eid, 'scene_id': scene, 'models': {}}
               for eid, scene in zip(ids, m['scene_ids'])]
    for label in labels:
        root = a.root / 'runlogs' / ('positive_matched_precision_val_unseen'
               if label == 'positive_initial_sft' else 'positive_extra_val_unseen')
        if not (root / (label + '.completed')).exists() or (root / (label + '.failed')).exists():
            raise ValueError('raw label is incomplete')
        for shard in range(4):
            folder = root / label / ('shard_%02d' % shard)
            selected = list(range(shard, len(ids), 4))
            if {x.name for x in (folder / 'log').glob('stats_*_0.json')} != {'stats_' + ids[i] + '_0.json' for i in selected}:
                raise ValueError('raw stats coverage differs')
            if {x.name for x in (folder / 'extra_info').glob('info_*_0.json')} != {'info_' + ids[i] + '_0.json' for i in selected}:
                raise ValueError('assistant-turn output coverage differs')
            for i in selected:
                eid = ids[i]
                stats_path = folder / 'log' / ('stats_' + eid + '_0.json')
                info_path = folder / 'extra_info' / ('info_' + eid + '_0.json')
                stats = json.loads(stats_path.read_text())
                if str(stats['id']) != eid or stats['early_stop_reason'] == 'inference_error':
                    raise ValueError('raw internal episode ID or inference error differs')
                src = source_rows[i]['models'][label]
                for key in ('success', 'spl', 'path_length', 'distance_to_goal', 'early_stop_reason'):
                    if stats[key] != src[key]:
                        raise ValueError('raw/compact metric differs: ' + key)
                conv = json.loads(info_path.read_text())['conversations']
                if not 1 <= len(conv) <= 12:
                    raise ValueError('saved generated turn coverage differs')
                actions = []
                for turn in conv:
                    if turn['role'] != 'assistant' or len(turn['content']) != 1 or turn['content'][0]['type'] != 'text':
                        raise ValueError('saved assistant schema differs')
                    parsed = action_parser.extract_multi_result(turn['content'][0]['text'])
                    actions.append([item[0] for item in parsed])
                    if not actions[-1] or any(v not in (None, 0, 1, 2, 3) for v in actions[-1]):
                        raise ValueError('parsed action ID invalid')
                records[i]['models'][label] = {k: src[k] for k in ('success', 'path_length', 'distance_to_goal', 'early_stop_reason')}
                records[i]['models'][label].update({'saved_assistant_turns': len(conv),
                    'parsed_action_ids_by_turn': actions,
                    'raw_stats_sha256': sha(stats_path), 'raw_assistant_info_sha256': sha(info_path)})
    summaries = {label: description([r['models'][label] for r in records]) for label in labels}
    pairs = {}
    for label in labels[1:]:
        lost = [r for r in records if r['models']['positive_initial_sft']['success'] == 1
                and r['models'][label]['success'] == 0]
        pairs[label] = {'sft_success_trained_failure': len(lost),
                       'trained_last_turn_first_stop_proposal': sum(r['models'][label]['parsed_action_ids_by_turn'][-1][0] == 0 for r in lost),
                       'trained_median_saved_turns': statistics.median(r['models'][label]['saved_assistant_turns'] for r in lost),
                       'sft_median_saved_turns': statistics.median(r['models']['positive_initial_sft']['saved_assistant_turns'] for r in lost)}
    report = {'schema': 'positive_posthoc_terminal_proposal_diagnostic_v1', 'episodes_per_model': 1839,
              'models': 7, 'manifest_sha256': MANIFEST_SHA, 'source_compact_sha256': COMPACT_SHA,
              'evaluator_parser_sha256': PARSER_SHA, 'exporter_sha256': sha(Path(__file__)),
              'raw_internal_ids_checked': 12873, 'source_assistant_outputs_checked': 12873,
              'inference_errors': 0, 'model_calls': 0, 'summaries': summaries, 'paired_sft_lost_successes': pairs,
              'scope': ['Post-hoc description of completed evaluation records; does not change any training, gate, selection or metric.',
                        'Parsed generated action proposals, not a separately recorded low-level executed-action trace.',
                        'Unclassified early_stop_reason is not on its own treated as STOP.',
                        'No causal identification, deployable semantic verification or new navigation benefit.',
                        'Existing adaptive-split, shared-SFT-decode and configured-seed limitations remain.']}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir / 'episodes.jsonl'
    with out.open('x') as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    report['diagnostic_compact_sha256'] = sha(out)
    with (a.output_dir / 'report.json').open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'episodes': 1839, 'model_calls': 0, 'summaries': {
        k: {n: v[n] for n in ('successes', 'median_saved_assistant_turns',
             'failure_last_turn_first_stop_proposal', 'failure_first_stop_proposal_by_three_saved_turns')}
        for k, v in summaries.items()}}))


if __name__ == '__main__':
    main()
