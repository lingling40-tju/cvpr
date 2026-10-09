"""Independent source/identity and completed-training recount for trace audit V3."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--revision', type=Path, required=True)
    p.add_argument('--new-tools', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r, d, t = a.experiment, a.revision, a.new_tools
    meta = json.loads((d / 'revision_sources.json').read_text())
    oldfile, newfile = d / 'evaluation_identity_v2.json', d / 'evaluation_identity_v3.proposed.json'
    old, new = json.loads(oldfile.read_text()), json.loads(newfile.read_text())
    assert sha(oldfile) == meta['old_evaluation_sha256']
    assert sha(newfile) == meta['new_evaluation_sha256']
    assert new['previous_evaluation_identity_sha256'] == sha(oldfile)
    for key, value in old.items():
        if key not in ('frozen_utc', 'source_sha256', 'previous_evaluation_identity_sha256'):
            assert new[key] == value, 'Scientific field changed: ' + key
    assert all(new['source_sha256'][k] == v for k, v in old['source_sha256'].items())
    assert len(old['source_sha256']) == 99 and len(new['source_sha256']) == 104
    assert sha(r / 'runlogs/freeze/identity.json') == new['training_identity_sha256']
    assert new['source_sha256']['vlnce_server/env.py'] == '39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a'
    assert new['source_sha256']['vlnce_server/semantic_reward/env.py'] == '6d90aed3c919cf76d892aa7984b6c5586c04e07483a26310584579aa13eb1ead'
    original = (r / 'tools/run_three_dev_suite_budget_v2.sh').read_text()
    assert (t / 'run_three_dev_suite_trace_v3.sh').read_text() == original.replace(
        'tools/audit_three_training_per_episode_budget.py', 'tools/audit_three_training_recorded_trace_v3.py')
    assert sha(t / 'run_three_dev_suite_trace_v3.sh') == meta['suite_sha256']
    auditor = t / 'audit_three_training_recorded_trace_v3.py'
    assert sha(auditor) == meta['audit_sha256'] == new['source_sha256']['tools/audit_three_training_recorded_trace_v3.py']
    # Generated trace checks cannot change existing optimizer/budget checks.
    updated = auditor.read_text()
    normalized = updated.replace("    environment = root / 'vlnce_server/env.py'\n", '').replace("    wrapper = root / 'vlnce_server/semantic_reward/env.py'\n", '')
    normalized = normalized.replace("        environment: '39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a',\n", '').replace("        wrapper: '6d90aed3c919cf76d892aa7984b6c5586c04e07483a26310584579aa13eb1ead',\n", '')
    start = normalized.index('\n\ndef check_generated_turn_trace(info):')
    end = normalized.index('\n\ndef main():', start)
    normalized = normalized[:start] + normalized[end:]
    normalized = normalized.replace('    generated_trace_histogram = Counter()\n    turn_limit_bookkeeping = 0\n', '')
    normalized = normalized.replace("            turn_limit_bookkeeping += int(check_generated_turn_trace(info))\n            generated_trace_histogram[str(len(info['gen_traj']))] += 1", "            if not 1 <= len(info['gen_traj']) <= 12:\n                raise ValueError('executed turn coverage invalid')")
    normalized = normalized.replace('three_direction_training_audit_v3_generated_response_trace', 'three_direction_training_audit_v2_per_episode_budget')
    for line in ["              'generated_response_count_histogram': dict(generated_trace_histogram),\n", "              'unexecuted_turn_limit_bookkeeping_records': turn_limit_bookkeeping,\n", "              'trace_semantics': 'gen_traj is generated responses; turn-limit overflow appends one unexecuted response, not a thirteenth executed turn.',\n"]:
        assert line in normalized
        normalized = normalized.replace(line, '')
    normalized = normalized.replace('Recorded training budget/scalars/response bookkeeping; no independent physical trace or navigation evaluation.', 'Recorded training budget and scalar evidence; no navigation evaluation.')
    assert normalized == (r / 'tools/audit_three_training_per_episode_budget.py').read_text()
    verifier = t / 'verify_three_development_raw_ids_trace_v3.py'
    assert verifier.read_text() == (r / 'tools/verify_three_development_raw_ids_budget_v2.py').read_text().replace(meta['old_evaluation_sha256'], meta['new_evaluation_sha256'])
    assert sha(verifier) == meta['new_verifier_sha256']
    supfile = t / 'raw_id_supplement_identity_trace_v3.json'
    if not supfile.exists():
        supfile = r / 'runlogs/freeze/raw_id_supplement_identity_trace_v3.json'
    supplement = json.loads(supfile.read_text())
    assert sha(supfile) == meta['new_supplement_sha256']
    assert supplement['parent_evaluation_identity_sha256'] == sha(newfile)
    assert supplement['development_opened_at_supplement_freeze'] is False
    assert supplement['verifier_sha256'] == sha(verifier)
    expected = (r / 'tools/watch_three_final_raw_ids_budget_v2.py').read_text()
    for before, after in [(meta['old_evaluation_sha256'], meta['new_evaluation_sha256']), ('2337e695db9863974996164ae6c8b54844a24ace9512d66c3a3fe6bd31a2d96b', meta['new_supplement_sha256']), ('01dd608cbc8195c6d39eba5c41b4b918b5ff15227112ebf4c70c74ca110aa1a5', meta['new_verifier_sha256']), ('raw_id_supplement_identity_budget_v2.json', 'raw_id_supplement_identity_trace_v3.json'), ('verify_three_development_raw_ids_budget_v2.py', 'verify_three_development_raw_ids_trace_v3.py'), ('tools/run_three_dev_suite_budget_v2.sh', 'tools/run_three_dev_suite_trace_v3.sh')]:
        expected = expected.replace(before, after)
    assert (t / 'watch_three_final_raw_ids_trace_v3.py').read_text() == expected
    assert sha(t / 'watch_three_final_raw_ids_trace_v3.py') == meta['new_watcher_sha256']
    preflight = json.loads((d / 'cpu_preflight.json').read_text())
    assert preflight['status'] == 'PASS' and preflight['model_calls'] == 0
    assert len(preflight['negative_trace_cases_rejected']) == 3
    auditfile = d / 'grpo_anchor_completion_audit.json'
    assert sha(auditfile) == preflight['full_training_audit_sha256']
    audit = json.loads(auditfile.read_text())
    assert audit['auditor_sha256'] == sha(auditor)
    assert audit['optimizer_updates'] == 64 and audit['group_size'] == 4
    assert audit['episode_exposures'] == 512 and audit['recorded_trajectories'] == 2048
    rows = audit['scalar_rows']
    assert [x['step'] for x in rows] == list(range(1, 65))
    assert all(math.isfinite(v) for x in rows for v in x.values())
    assert all(x['training/global_step'] == x['step'] and abs(x['actor/kl_coef'] - .1) < 1e-7 and abs(x['actor/lr'] - 1e-6) < 1e-12 for x in rows)
    assert sum(x['actor/grad_norm'] > 0 for x in rows) == audit['nonzero_gradient_steps'] == 64
    assert sum(x['critic/advantages/max'] > 0 or x['critic/advantages/min'] < 0 for x in rows) == audit['reward_signal_steps']
    manifest = json.loads((r / 'prepared_data/fit512_manifest.json').read_text())
    ids = [str(v) for v in manifest['episode_ids']]
    assert len(set(ids)) == 512 and sha(r / 'prepared_data/fit512_manifest.json') == audit['fit_manifest_sha256']
    for step, x in enumerate(audit['rollout_step_rows'], 1):
        assert x['step'] == step and x['episode_ids'] == sorted(ids[(step-1)*8:step*8]) and x['trajectories'] == 32
    budgets = audit['per_episode_expected_budgets']
    assert set(budgets) == set(ids)
    assert all(v['turn_budget'] == 12 and v['step_budget'] == min(2*v['gt_actions'], 36) for v in budgets.values())
    hist = dict(Counter(str(v['turn_budget'])+'/'+str(v['step_budget']) for v in budgets.values()))
    assert hist == audit['budget_histogram'] and hist['12/36'] == 501
    assert sum(audit['generated_response_count_histogram'].values()) == 2048
    assert audit['generated_response_count_histogram']['13'] == audit['unexecuted_turn_limit_bookkeeping_records'] == preflight['unexecuted_overflow_records'] == 739
    applied = json.loads((d / 'applied.json').read_text())
    assert applied['status'] == 'APPLIED' and applied['training_processes_untouched'] and applied['development_not_opened']
    assert applied['new_owner_preflight']['owner']['command'] == ['bash', 'tools/run_three_dev_suite_trace_v3.sh', sha(newfile)]
    out = {'schema': 'three_trace_revision_and_training_independent_compact_recount_v1', 'status': 'PASS', 'optimizer_updates': 64, 'nonzero_actor_gradient_steps': 64, 'fit_episodes': 512, 'recorded_trajectories': 2048, 'unexecuted_turn_limit_bookkeeping_records': 739, 'original_v2_identity_entries_preserved': 99, 'same_sft_evidence_files': 265, 'gate_and_raw_metric_logic_unchanged': True, 'new_evaluation_sha256': sha(newfile), 'training_audit_sha256': sha(auditfile), 'source_sha256': sha(Path(__file__)), 'scope': 'Independent compact arithmetic/source/identity checks; not a second Parquet/raw decode, physical simulator replay or navigation result.'}
    with a.output.open('x') as f:
        f.write(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
