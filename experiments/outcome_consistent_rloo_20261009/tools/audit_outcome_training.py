"""Raw-source CPU audit for the outcome-consistent RLOO n=4 pilot."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--updates', type=int, choices=(2, 64), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    identity_path = root / 'runlogs/freeze/identity.json'
    identity = json.loads(identity_path.read_text())
    for rel, digest in identity['source_sha256'].items():
        if sha(root / rel) != digest:
            raise ValueError('frozen source changed: ' + rel)
    identity_payload = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    canonical_identity = json.dumps(identity_payload, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical_identity).hexdigest() != identity['identity_sha256']:
        raise ValueError('frozen identity self-hash differs')

    label = 'td_outcome_consistent_rloo_n4_seed11'
    run = root / 'runlogs' / label
    checkpoint = root / 'verl_checkpoints' / label
    step_max = args.updates
    if args.updates == 64:
        if not (run / 'completed').exists() or (run / 'failed').exists():
            raise ValueError('full training incomplete or failed')
        ck = checkpoint / 'global_step_64'
    else:
        if not (run / 'smoke.completed').exists() or (run / 'failed').exists():
            raise ValueError('two-update smoke incomplete or failed')
        ck = checkpoint / 'global_step_2'
    if not (ck / 'actor/huggingface/config.json').is_file():
        raise ValueError('expected checkpoint missing')

    tags = ('actor/grad_norm', 'actor/kl_loss', 'actor/kl_coef', 'actor/lr',
            'actor/pg_loss', 'critic/advantages/max', 'critic/advantages/min',
            'critic/score/max', 'training/global_step')
    values = {tag: {} for tag in tags}
    event_hashes = {}
    for event in sorted((run / 'tensorboard').glob('events*')):
        event_hashes[event.name] = sha(event)
        accumulator = EventAccumulator(str(event), size_guidance={'scalars': 0})
        accumulator.Reload()
        available = accumulator.Tags().get('scalars', [])
        for tag in tags:
            if tag not in available:
                continue
            for row in accumulator.Scalars(tag):
                if row.step in values[tag] or not math.isfinite(row.value):
                    raise ValueError('duplicate or nonfinite scalar ' + tag)
                values[tag][row.step] = row.value
    expected_steps = set(range(1, step_max + 1))
    if any(set(value) != expected_steps for value in values.values()):
        raise ValueError('TensorBoard step coverage differs')
    for step in expected_steps:
        if values['training/global_step'][step] != step:
            raise ValueError('optimizer step numbering differs')
        if values['actor/grad_norm'][step] <= 0:
            raise ValueError('actor gradient is zero at step ' + str(step))
        if abs(values['actor/kl_coef'][step] - .1) > 1e-7:
            raise ValueError('reference KL coefficient differs')
        if abs(values['actor/lr'][step] - 1e-6) > 1e-12:
            raise ValueError('learning rate differs')
        if not 0 <= values['critic/score/max'][step] <= 20.001:
            raise ValueError('terminal reward range differs')
    signal_steps = sum(values['critic/advantages/max'][step] > 0 or
                       values['critic/advantages/min'][step] < 0
                       for step in expected_steps)
    if not signal_steps:
        raise ValueError('no actor advantage signal')

    manifest_path = root / 'prepared_data/fit512_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    fit_ids = [str(value) for value in manifest['episode_ids']]
    if len(fit_ids) != 512 or len(set(fit_ids)) != 512:
        raise ValueError('fit manifest is not 512 unique rows')
    parquet = pd.read_parquet(root / 'prepared_data/fit512.parquet')
    extra_info = list(parquet['extra_info'])
    if [str(item['episode_id']) for item in extra_info] != fit_ids:
        raise ValueError('Parquet and fit manifest order differ')
    expected_budgets = {str(item['episode_id']):
                         {'step_budget': int(min(2 * len(item['gt_actions']), 36)), 'turn_budget': 12}
                         for item in extra_info}
    if len(expected_budgets) != 512:
        raise ValueError('fit budget map differs')

    raw_path = checkpoint / 'rollout.jsonl'
    step_rows = {}
    trace_histogram = Counter()
    terminal_counts = Counter()
    forced_counts = Counter()
    success_only_violations = 0
    for line in raw_path.open():
        rollout = json.loads(line)
        step = rollout['step']
        if step not in expected_steps or step in step_rows:
            raise ValueError('duplicate or invalid raw optimizer step')
        infos = rollout['info']
        count = Counter(str(info['episode_id']) for info in infos)
        expected_ids = fit_ids[(step - 1) * 8:step * 8]
        if len(infos) != 32 or count != Counter({eid: 4 for eid in expected_ids}):
            raise ValueError('n=4 group or batch membership differs')
        for info in infos:
            eid = str(info['episode_id'])
            budget = expected_budgets[eid]
            if info['step_budget'] != budget['step_budget'] or info['turn_budget'] != 12:
                raise ValueError('per-row trajectory budget differs: ' + eid)
            trace = info['gen_traj']
            trace_histogram[str(len(trace))] += 1
            if not info.get('done'):
                raise ValueError('rollout info is not terminal')
            forced = info.get('forced_stop_reason')
            if forced not in (None, 'turn_budget', 'step_budget'):
                raise ValueError('unknown forced STOP source')
            if forced:
                if trace[-1].get('forced_stop_reason') != forced:
                    raise ValueError('final generated response lacks forced STOP source: ' + eid)
                if any(item.get('forced_stop_reason') for item in trace[:-1]):
                    raise ValueError('forced STOP source appears before final generated response: ' + eid)
                forced_counts[forced] += 1
                if bool(info['task_success']) != bool(info['orcale_success']):
                    raise ValueError('forced STOP outcome disagrees with evaluation oracle: ' + eid)
                if info['end_reason'] not in ('successfully reached the goal.', 'stopped but goal not reached.'):
                    raise ValueError('forced STOP has non-STOP terminal reason: ' + eid)
                if forced == 'turn_budget':
                    if len(trace) != info['turn_budget'] + 1 or info['executed_actions']:
                        raise ValueError('turn-overflow trace does not terminate in a non-executed forced STOP')
                    if trace[-1].get('executed_actions'):
                        raise ValueError('turn-overflow final response was executed')
                elif len(trace) > info['turn_budget']:
                    raise ValueError('step-budget truncation exceeded the response-turn budget')
            elif len(trace) > info['turn_budget']:
                raise ValueError('unexpected generated response overflow')

            reward = float(info['total_reward'])
            components = info.get('reward_components', {})
            ndtw_credit = float(components.get('ndtw_reward', 0.0))
            if not math.isfinite(reward) or not 0 <= reward <= 20.001:
                raise ValueError('invalid total terminal reward')
            if not math.isfinite(ndtw_credit) or not 0 <= ndtw_credit <= 5.001:
                raise ValueError('invalid path credit')
            if not bool(info['task_success']) and ndtw_credit != 0:
                success_only_violations += 1
            terminal_counts['success' if info['task_success'] else 'failure'] += 1
        step_rows[step] = {'step': step, 'episode_ids': sorted(count), 'trajectory_count': len(infos)}
    if set(step_rows) != expected_steps:
        raise ValueError('raw rollout update coverage differs')
    if success_only_violations:
        raise ValueError('failed trajectories received success-conditioned nDTW credit')

    report = {
        'schema': 'outcome_consistent_rloo_training_audit_v1',
        'pilot_identity_sha256': identity['identity_sha256'],
        'updates_checked': step_max,
        'configured_seed': 11,
        'fit_episodes': 512,
        'episode_exposures': 8 * step_max,
        'recorded_trajectories': 32 * step_max,
        'group_size': 4,
        'batch_rows': 8,
        'nonzero_actor_gradient_steps': step_max,
        'reward_signal_steps': signal_steps,
        'terminal_outcomes': dict(terminal_counts),
        'forced_stop_counts': dict(forced_counts),
        'success_conditioned_ndtw_violations': success_only_violations,
        'generated_response_count_histogram': dict(trace_histogram),
        'scalar_rows': [{'step': step, **{tag: values[tag][step] for tag in tags}}
                        for step in sorted(expected_steps)],
        'rollout_step_rows': [step_rows[step] for step in sorted(expected_steps)],
        'tensorboard_event_sha256': event_hashes,
        'rollout_jsonl_sha256': sha(raw_path),
        'fit_manifest_sha256': sha(manifest_path),
        'fit_parquet_sha256': sha(root / 'prepared_data/fit512.parquet'),
        'train_log_sha256': sha(run / 'train.log'),
        'checkpoint_config_sha256': sha(ck / 'actor/huggingface/config.json'),
        'auditor_sha256': sha(Path(__file__)),
        'frozen_source_sha256': identity['source_sha256'],
        'scope': 'Raw-source training/scalar/terminal outcome audit; no claim of navigation benefit or semantic truth.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('schema', 'updates_checked', 'nonzero_actor_gradient_steps',
                                               'terminal_outcomes', 'forced_stop_counts',
                                               'success_conditioned_ndtw_violations')}))


if __name__ == '__main__':
    main()
