"""Combine independently recounted fixed development comparisons into the original gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--identity-sha', required=True)
    a = p.parse_args()
    descriptor = a.root / 'runlogs/freeze/evaluation_identity.json'
    if sha(descriptor) != a.identity_sha:
        raise ValueError('evaluation freeze differs')
    state = a.root / 'runlogs/development_suite'
    result = a.root / 'runlogs/development256'
    comparisons = {}
    evidence = {}
    for method in ('grpo_anchor', 'turn_rloo', 'srgpo'):
        for reference in ('sft', 'grpo'):
            if method == 'grpo_anchor' and reference == 'grpo':
                continue
            name = method + '_vs_' + reference
            report = json.loads((state / (name + '.json')).read_text())
            recount = json.loads((state / (name + '.independent.json')).read_text())
            expected_control = 'positive_initial_sft' if reference == 'sft' else 'td_grpo_anchor_n4_seed11'
            if report['control'] != expected_control or report['candidate'] != 'td_' + method + '_n4_seed11':
                raise ValueError('comparison identity differs')
            for metric in ('paired_sr_points', 'paired_spl_points'):
                if not math.isclose(report[metric], recount[metric], abs_tol=1e-10, rel_tol=0):
                    raise ValueError('independent paired metric differs')
            if recount['episodes'] != 256 or recount['scenes'] != 8 or recount['inference_errors'] != 0 or not recount['bootstrap_recount_agrees']:
                raise ValueError('independent coverage differs')
            if report['compact_sha256'] != sha(state / (name + '.jsonl')):
                raise ValueError('compact source changed')
            comparisons[name] = report
            for ext in ('.json', '.jsonl', '.independent.json'):
                evidence[name + ext] = sha(state / (name + ext))
    gates = {}
    for method in ('grpo_anchor', 'turn_rloo', 'srgpo'):
        rs = comparisons[method + '_vs_sft']
        sft_pass = rs['paired_sr_points'] >= 2 and rs['paired_spl_points'] >= 2
        rg = comparisons.get(method + '_vs_grpo')
        grpo_pass = rg is None or (rg['paired_sr_points'] >= 2 and rg['paired_spl_points'] >= 2)
        gates[method] = {'pass_vs_sft': sft_pass,
                         'pass_vs_grpo_required': method != 'grpo_anchor',
                         'pass_vs_grpo': grpo_pass if rg else None,
                         'eligible_for_scale': sft_pass and grpo_pass}
    report = {'schema': 'three_direction_development_report_v1', 'episodes_per_model': 256,
              'scenes': 8, 'configured_training_seed': 11, 'group_size': 4,
              'total_updates_including_smoke': 64, 'smoke_updates': 2,
              'inference_dtype': 'float16', 'sft_is_one_reused_decode_reference': True,
              'evaluation_identity_sha256': a.identity_sha,
              'requires_each_metric_pp_at_least': 2.0, 'gates': gates,
              'comparisons': comparisons, 'pair_evidence_sha256': evidence,
              'reserved_opened': False, 'val_unseen_opened': False,
              'limits': ['Adaptively reused train-scene development screen; exploratory.',
                         'One configured training seed; rollout-stream independence not established.',
                         'RLOO/SRGPO process feedback is privileged simulator geometry.',
                         'SRGPO-style composite includes population outcome normalization; a gain alone does not isolate process grouping.',
                         'Training is a common two-update smoke followed by 62 resumed updates.']}
    for label in ('positive_initial_sft', 'td_grpo_anchor_n4_seed11', 'td_turn_rloo_n4_seed11', 'td_srgpo_n4_seed11'):
        dtype = json.loads((result / (label + '.dtype.json')).read_text())
        expected_checkpoint = (Path('/Knowin/foundation/haozhiwang/whz/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn')
                               if label == 'positive_initial_sft' else
                               a.root / 'verl_checkpoints' / label / 'global_step_64/actor/huggingface')
        if dtype['actual_engine_dtype'] != 'float16' or Path(dtype['checkpoint']) != expected_checkpoint or dtype['engine_seed'] != 11:
            raise ValueError('actual inference precision/checkpoint differs')
        if sha(Path(dtype['log'])) != dtype['log_snapshot_sha256'] or sha(expected_checkpoint / 'config.json') != dtype['checkpoint_config_sha256']:
            raise ValueError('completed inference identity changed')
        evidence[label + '.validated.json'] = sha(result / (label + '.validated.json'))
        evidence[label + '.dtype.json'] = sha(result / (label + '.dtype.json'))
    out = state / 'three_direction_report.json'
    if out.exists() and json.loads(out.read_text()) != report:
        raise ValueError('existing aggregate report differs')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'gates': gates}))


if __name__ == '__main__':
    main()
