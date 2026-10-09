"""CPU recount of a completed 64-update n=4 pilot, including its 2 smoke updates."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--method', choices=('grpo_anchor', 'turn_rloo', 'srgpo'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    label = 'td_' + a.method + '_n4_seed11'
    run = a.root / 'runlogs' / label
    ck = a.root / 'verl_checkpoints' / label
    if not (run / 'completed').exists() or (run / 'failed').exists():
        raise ValueError('64-step training is incomplete or failed')
    if not (ck / 'global_step_64/actor/huggingface/config.json').exists():
        raise ValueError('step-64 checkpoint missing')
    tags = ('actor/grad_norm', 'actor/kl_loss', 'actor/kl_coef', 'actor/lr',
            'actor/pg_loss', 'critic/advantages/max', 'critic/advantages/min',
            'critic/score/max', 'training/global_step')
    values = {t: {} for t in tags}
    event_sha = {}
    for f in sorted((run / 'tensorboard').glob('events*')):
        event_sha[f.name] = sha(f)
        ea = EventAccumulator(str(f), size_guidance={'scalars': 0})
        ea.Reload()
        for tag in tags:
            if tag not in ea.Tags().get('scalars', []):
                continue
            for v in ea.Scalars(tag):
                if v.step in values[tag]:
                    raise ValueError('duplicate scalar step: ' + tag + ' ' + str(v.step))
                if not math.isfinite(v.value):
                    raise ValueError('nonfinite scalar')
                values[tag][v.step] = v.value
    expected = set(range(1, 65))
    if any(set(v) != expected for v in values.values()):
        raise ValueError('TensorBoard optimizer coverage differs from 1..64')
    for st in expected:
        if values['training/global_step'][st] != st or values['actor/grad_norm'][st] < 0:
            raise ValueError('invalid optimizer step or gradient')
        if abs(values['actor/kl_coef'][st] - .1) > 1e-7:
            raise ValueError('fixed KL coefficient differs')
        if abs(values['actor/lr'][st] - 1e-6) > 1e-12:
            raise ValueError('fixed optimizer learning rate differs')
        if not 0 <= values['critic/score/max'][st] <= 20.001:
            raise ValueError('terminal score differs from frozen 0..20 range')
    gradients = sum(v > 0 for v in values['actor/grad_norm'].values())
    signals = sum(values['critic/advantages/max'][s] > 0 or
                  values['critic/advantages/min'][s] < 0 for s in expected)
    if gradients == 0 or signals == 0:
        raise ValueError('no actor gradient or reward advantage signal')
    manifest = a.root / 'prepared_data/fit512_manifest.json'
    fit = [str(x) for x in json.loads(manifest.read_text())['episode_ids']]
    if len(fit) != 512 or len(set(fit)) != 512:
        raise ValueError('fit manifest coverage differs')
    raw = ck / 'rollout.jsonl'
    steps = {}
    for line in raw.open():
        x = json.loads(line)
        st = x['step']
        if st in steps or st not in expected:
            raise ValueError('duplicate or invalid raw rollout step')
        infos = x['info']
        count = Counter(str(i['episode_id']) for i in infos)
        wanted = fit[(st - 1) * 8:st * 8]
        if len(infos) != 32 or count != Counter({eid: 4 for eid in wanted}):
            raise ValueError('raw rollout n4/batch8/fit-order budget differs')
        for info in infos:
            reward = info['total_reward']
            if not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 20.001:
                raise ValueError('raw terminal reward invalid')
            if info['step_budget'] != 36 or info['turn_budget'] != 12:
                raise ValueError('raw action or turn budget differs')
            if not 1 <= len(info['gen_traj']) <= 12:
                raise ValueError('executed turn coverage invalid')
        steps[st] = {'step': st, 'episode_ids': sorted(count), 'trajectories': len(infos)}
    if set(steps) != expected:
        raise ValueError('raw rollout step coverage differs')
    report = {'schema': 'three_direction_training_audit_v1', 'method': a.method,
              'configured_seed': 11, 'optimizer_updates': 64, 'smoke_updates_in_total': 2,
              'continuation_updates': 62, 'fit_episodes': 512, 'episode_exposures': 512,
              'recorded_trajectories': 2048, 'group_size': 4, 'batch_rows': 8,
              'nonzero_gradient_steps': gradients, 'reward_signal_steps': signals,
              'scalar_precision': 'TensorBoard stored float32',
              'scalar_rows': [{'step': s, **{t: values[t][s] for t in tags}} for s in sorted(expected)],
              'rollout_step_rows': [steps[s] for s in sorted(steps)],
              'event_sha256': event_sha, 'rollout_sha256': sha(raw),
              'fit_manifest_sha256': sha(manifest), 'train_log_sha256': sha(run / 'train.log'),
              'checkpoint_config_sha256': sha(ck / 'global_step_64/actor/huggingface/config.json'),
              'auditor_sha256': sha(Path(__file__)),
              'scope': 'Recorded training budget and scalar evidence; no navigation evaluation.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        if json.loads(a.output.read_text()) != report:
            raise ValueError('existing training audit differs')
    else:
        with a.output.open('x') as f:
            f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('method', 'optimizer_updates', 'nonzero_gradient_steps', 'recorded_trajectories')}))


if __name__ == '__main__':
    main()
