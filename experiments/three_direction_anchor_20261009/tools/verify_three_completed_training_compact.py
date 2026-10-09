"""Independent compact training recount; no raw rollout or TensorBoard import."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--method', choices=('grpo_anchor', 'turn_rloo', 'srgpo'), required=True)
    p.add_argument('--expected-audit-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = a.experiment
    require(sha(a.audit) == a.expected_audit_sha, 'Transferred audit SHA differs')
    audit = json.loads(a.audit.read_text())
    freeze_file = r / 'runlogs/freeze/evaluation_identity.json'
    freeze = json.loads(freeze_file.read_text())
    require(sha(freeze_file) == '0ad45c7e518166fd8ffce502b7bd6354b201cce12729bd9e127e33ff9a319a99', 'Evaluation identity differs')
    require(sha(r / 'runlogs/freeze/identity.json') == freeze['training_identity_sha256'], 'Parent training identity differs')
    require(audit['schema'] == 'three_direction_training_audit_v3_generated_response_trace' and audit['method'] == a.method, 'Audit schema or method differs')
    src = r / 'tools/audit_three_training_recorded_trace_v3.py'
    require(audit['auditor_sha256'] == sha(src) == freeze['source_sha256']['tools/audit_three_training_recorded_trace_v3.py'], 'Raw auditor source differs')
    prior = json.loads((r / 'runlogs/evaluation_trace_audit_revision20261009/grpo_anchor_completion_audit.json').read_text())
    require(audit['budget_source_sha256'] == prior['budget_source_sha256'], 'Frozen budget-source provenance differs from completed GRPO')
    checked_local = []
    for name, digest in audit['budget_source_sha256'].items():
        path = r / name
        if path.exists():
            require(sha(path) == digest, 'Local available source differs: ' + name)
            checked_local.append(name)
        if name in freeze['source_sha256']:
            require(freeze['source_sha256'][name] == digest, 'Evaluation-bound source differs: ' + name)
    for key, wanted in {'configured_seed': 11, 'optimizer_updates': 64, 'smoke_updates_in_total': 2, 'continuation_updates': 62, 'fit_episodes': 512, 'episode_exposures': 512, 'recorded_trajectories': 2048, 'group_size': 4, 'batch_rows': 8}.items():
        require(audit[key] == wanted, 'Frozen count differs: ' + key)
    manifest_file = r / 'prepared_data/fit512_manifest.json'
    require(sha(manifest_file) == audit['fit_manifest_sha256'], 'Manifest SHA differs')
    ids = [str(v) for v in json.loads(manifest_file.read_text())['episode_ids']]
    require(len(ids) == len(set(ids)) == 512, 'Manifest IDs are not 512 unique members')
    tags = {'actor/grad_norm', 'actor/kl_loss', 'actor/kl_coef', 'actor/lr', 'actor/pg_loss', 'critic/advantages/max', 'critic/advantages/min', 'critic/score/max', 'training/global_step', 'step'}
    rows = audit['scalar_rows']
    require([row['step'] for row in rows] == list(range(1, 65)), 'Optimizer coverage differs')
    for row in rows:
        require(set(row) == tags, 'Stored scalar fields differ')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in row.values()), 'Scalar is nonfinite or nonnumeric')
        require(row['training/global_step'] == row['step'] and row['actor/grad_norm'] >= 0, 'Global step or gradient differs')
        require(abs(row['actor/kl_coef'] - .1) <= 1e-7 and abs(row['actor/lr'] - 1e-6) <= 1e-12, 'KL or learning rate differs')
        require(0 <= row['critic/score/max'] <= 20.001, 'Stored score range differs')
    gradients = sum(row['actor/grad_norm'] > 0 for row in rows)
    signals = sum(row['critic/advantages/max'] > 0 or row['critic/advantages/min'] < 0 for row in rows)
    require(gradients == audit['nonzero_gradient_steps'] and gradients > 0, 'Gradient recount differs')
    require(signals == audit['reward_signal_steps'] and signals > 0, 'Advantage recount differs')
    rollout = audit['rollout_step_rows']
    require(len(rollout) == 64, 'Recorded rollout row count differs')
    exposure = Counter()
    for st, row in enumerate(rollout, 1):
        wanted = sorted(ids[(st - 1) * 8:st * 8])
        require(row == {'step': st, 'episode_ids': wanted, 'trajectories': 32}, 'Recorded rollout membership differs')
        exposure.update(row['episode_ids'])
    require(exposure == Counter(ids), 'Fit exposure membership differs')
    require(sum(row['trajectories'] for row in rollout) == 2048, 'Recorded trajectory count differs')
    budgets = audit['per_episode_expected_budgets']
    require(set(budgets) == set(ids) and budgets == prior['per_episode_expected_budgets'], 'Episode budget members or original row lengths differ')
    require(all(v['turn_budget'] == 12 and type(v['gt_actions']) is int and v['gt_actions'] >= 1 and v['step_budget'] == min(2 * v['gt_actions'], 36) for v in budgets.values()), 'Per-episode budget formula differs')
    histogram = dict(Counter(str(v['turn_budget']) + '/' + str(v['step_budget']) for v in budgets.values()))
    require(histogram == audit['budget_histogram'] and histogram['12/36'] == 501, 'Budget histogram differs')
    traces = audit['generated_response_count_histogram']
    require(all(str(int(k)) == k and 1 <= int(k) <= 13 and type(v) is int and v >= 0 for k, v in traces.items()), 'Generated-response histogram invalid')
    require(sum(traces.values()) == 2048 and traces.get('13', 0) == audit['unexecuted_turn_limit_bookkeeping_records'], 'Response bookkeeping recount differs')
    require(audit['scalar_precision'] == 'TensorBoard stored float32', 'Stored scalar precision differs')
    out = {'schema': 'three_completed_training_independent_compact_recount_v1', 'status': 'PASS', 'method': a.method, 'optimizer_updates': 64, 'nonzero_gradient_steps': gradients, 'reward_signal_steps': signals, 'fit_episodes': 512, 'episode_exposures': sum(exposure.values()), 'recorded_trajectories': 2048, 'unexecuted_turn_limit_bookkeeping_records': traces.get('13', 0), 'audit_sha256': sha(a.audit), 'evaluation_identity_sha256': sha(freeze_file), 'manifest_sha256': sha(manifest_file), 'source_sha256': sha(Path(__file__)), 'available_local_budget_sources_rehashed': checked_local, 'scope': 'Independent compact arithmetic, membership and source-identity checks. Does not independently decode raw rollout/TensorBoard/Parquet, rehash unavailable raw remote files, replay physical trajectories, validate weight tensor values or measure navigation.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        f.write(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
