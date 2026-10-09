"""Independent source/identity recount for the pre-inference budget-audit revision."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--new-tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.experiment
    revision = args.revision
    tools = args.new_tools
    meta = json.loads((revision / 'revision_sources.json').read_text())
    old_file = revision / 'evaluation_identity_v1.json'
    new_file = revision / 'evaluation_identity_v2.proposed.json'
    old = json.loads(old_file.read_text())
    new = json.loads(new_file.read_text())
    assert sha(old_file) == meta['old_evaluation_sha256']
    assert sha(new_file) == meta['new_evaluation_sha256']
    assert old['training_identity_sha256'] == new['training_identity_sha256']
    assert sha(base / 'runlogs/freeze/identity.json') == new['training_identity_sha256']
    for key, value in old.items():
        if key not in ('frozen_utc', 'source_sha256'):
            assert new[key] == value, 'Changed scientific field: ' + key
    assert set(new['source_sha256']) - set(old['source_sha256']) == {
        'tools/audit_three_training_per_episode_budget.py',
        'tools/run_three_dev_suite_budget_v2.sh',
        'runlogs/evaluation_audit_revision20261009/cpu_preflight.json'}
    assert all(new['source_sha256'][k] == v for k, v in old['source_sha256'].items())
    assert new['previous_evaluation_identity_sha256'] == sha(old_file)
    preflight = revision / 'cpu_preflight.json'
    p = json.loads(preflight.read_text())
    assert p['status'] == 'PASS' and p['model_calls'] == 0
    assert p['real_smoke_records_checked'] == 64
    assert sum(p['fit_budget_histogram'].values()) == 512
    assert p['fit_budget_histogram']['12/36'] == 501
    assert p['expected_budget'] == {'gt_actions': 13, 'step_budget': 26, 'turn_budget': 12}
    assert set(p['negative_cases_rejected']) == {'incorrect_all_36', 'incorrect_turn_budget', 'unknown_episode'}
    assert sha(preflight) == new['source_sha256']['runlogs/evaluation_audit_revision20261009/cpu_preflight.json']
    original = (base / 'tools/run_three_dev_suite.sh').read_text()
    revised = (tools / 'run_three_dev_suite_budget_v2.sh').read_text()
    assert original.count('tools/audit_three_training.py') == 1
    assert revised == original.replace('tools/audit_three_training.py', 'tools/audit_three_training_per_episode_budget.py')
    assert sha(tools / 'run_three_dev_suite_budget_v2.sh') == meta['suite_sha256'] == p['suite_sha256']
    original = (base / 'tools/audit_three_training.py').read_text()
    revised = (tools / 'audit_three_training_per_episode_budget.py').read_text()
    # Remove only the documented helper, per-row guard and provenance fields.
    normalized = revised.replace('\nimport pandas as pd\n', '')
    start = normalized.index('\n\ndef expected_episode_budgets(root):')
    end = normalized.index('\n\ndef main():', start)
    normalized = normalized[:start] + normalized[end:]
    normalized = normalized.replace("    budgets, budget_sources = expected_episode_budgets(a.root)\n", '')
    normalized = normalized.replace('            check_episode_budget(info, budgets)',
        "            if info['step_budget'] != 36 or info['turn_budget'] != 12:\n                raise ValueError('raw action or turn budget differs')")
    normalized = normalized.replace('three_direction_training_audit_v2_per_episode_budget', 'three_direction_training_audit_v1')
    for line in [
        "              'original_auditor_sha256': 'c4b001dd7f3ac97197f81599d98c289ac87d7248e86a16e9997b89b3bfe722af',\n",
        "              'budget_source_sha256': budget_sources,\n",
        "              'per_episode_expected_budgets': budgets,\n",
        "              'budget_histogram': dict(Counter(str(b['turn_budget']) + '/' + str(b['step_budget']) for b in budgets.values())),\n"]:
        assert line in normalized
        normalized = normalized.replace(line, '')
    assert normalized == original, 'Unrelated training-audit check changed'
    assert sha(tools / 'audit_three_training_per_episode_budget.py') == meta['audit_sha256'] == p['auditor_sha256']
    assert 'int(min(count * 2.0, 36))' in revised
    for rel, digest in p['source_sha256'].items():
        assert digest == json.loads((base / 'runlogs/freeze/identity.json').read_text())['source_sha256'][rel]
    verifier_file = tools / 'verify_three_development_raw_ids_budget_v2.py'
    expected = (base / 'tools/verify_three_development_raw_ids_20261009.py').read_text().replace(
        meta['old_evaluation_sha256'], meta['new_evaluation_sha256']).replace(
        'original frozen training/evaluation sources unchanged.',
        'Original training/inference/metric sources unchanged; transparent per-episode training-audit revision.')
    assert verifier_file.read_text() == expected
    assert sha(verifier_file) == meta['new_verifier_sha256']
    supplement = json.loads((tools / 'raw_id_supplement_identity_budget_v2.json').read_text())
    assert sha(tools / 'raw_id_supplement_identity_budget_v2.json') == meta['new_supplement_sha256']
    assert supplement['parent_evaluation_identity_sha256'] == meta['new_evaluation_sha256']
    assert supplement['development_opened_at_supplement_freeze'] is False
    assert supplement['verifier_sha256'] == sha(verifier_file)
    expected = (base / 'tools/watch_three_final_raw_ids_20261009.py').read_text()
    for old_value, new_value in [
        (meta['old_evaluation_sha256'], meta['new_evaluation_sha256']),
        ('9260ebbb393912269268e878297e390a0eafe3d62d3bf45457a522417ad93d39', meta['new_supplement_sha256']),
        ('547721f41b38298bfb83804bd6e8c7628d4ad0e806cbbcb42dac431fabb86908', meta['new_verifier_sha256']),
        ('raw_id_supplement_identity.json', 'raw_id_supplement_identity_budget_v2.json'),
        ('verify_three_development_raw_ids_20261009.py', 'verify_three_development_raw_ids_budget_v2.py'),
        ('tools/run_three_dev_suite.sh', 'tools/run_three_dev_suite_budget_v2.sh')]:
        expected = expected.replace(old_value, new_value)
    assert (tools / 'watch_three_final_raw_ids_budget_v2.py').read_text() == expected
    assert sha(tools / 'watch_three_final_raw_ids_budget_v2.py') == meta['new_watcher_sha256']
    applied = json.loads((revision / 'applied.json').read_text())
    assert applied['status'] == 'APPLIED' and applied['development_not_opened'] is True
    assert applied['training_processes_untouched'] is True
    assert applied['new_owner_preflight']['owner']['pid'] == applied['new_suite_pid']
    assert applied['new_owner_preflight']['owner']['command'] == ['bash', 'tools/run_three_dev_suite_budget_v2.sh', meta['new_evaluation_sha256']]
    out = {'schema': 'independent_three_budget_revision_recount_v1', 'status': 'PASS',
        'old_evaluation_sha256': sha(old_file), 'new_evaluation_sha256': sha(new_file),
        'unchanged_original_source_identities': len(old['source_sha256']),
        'unchanged_sft_raw_file_identities': len(old['sft_reference']['evidence_sha256']),
        'training_identity_unchanged': True, 'inference_gate_and_metric_sources_unchanged': True,
        'distinct_revised_audit_sha256': meta['audit_sha256'],
        'suite_only_changes_audit_invocation': True, 'raw_metric_recount_logic_unchanged': True,
        'negative_budget_tests_rejected': 3, 'new_suite_pid': applied['new_suite_pid'],
        'new_cpu_watcher_pid': applied['new_cpu_watcher_pid'],
        'source_sha256': sha(Path(__file__)),
        'scope': 'Source diff, frozen identity and recorded preflight recount; not independent Parquet decoding, new training completion, navigation or live GPU verification.'}
    with args.output.open('x') as f:
        f.write(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
