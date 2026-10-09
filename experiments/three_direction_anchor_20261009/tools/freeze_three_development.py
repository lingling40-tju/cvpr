"""Freeze development evaluation before candidate navigation outputs exist."""
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path('/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_direction_20261009')
OLD = ROOT.parent / 'ActiveVLN_positive_trajectory_20261006'
SFTROOT = OLD / 'runlogs/positive_matched_precision_development'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parent = ROOT / 'runlogs/freeze/identity.json'
    if sha(parent) != 'd0f5722d9ca93519d6555b6ecd9aa5173cc13a9b65f79679817fbb2e925e1d94':
        raise ValueError('parent training freeze differs')
    preflight_path = ROOT / 'runlogs/development_preflight/preflight.json'
    preflight = json.loads(preflight_path.read_text())
    if preflight['status'] != 'PASS' or preflight['model_calls'] != 0:
        raise ValueError('CPU preflight has not passed')
    result = ROOT / 'runlogs/development256'
    if result.exists() and any(result.glob('td_*')):
        raise ValueError('new development outputs were already opened')
    sources = ['tools/' + name for name in (
        'audit_three_training.py', 'verify_three_eval_freeze.py',
        'run_three_dev_model.sh', 'run_three_dev_suite.sh', 'aggregate_three_development.py',
        'eval_train_scene_subset.py', 'validate_train_label.py',
        'analyze_train_scene_pair.py', 'verify_positive_compact.py',
        'verify_positive_engine_dtype.py', 'preflight_three_pipeline.py',
        'freeze_three_development.py')]
    sources += ['prepared_data/development256.json',
                'prepared_data/positive_initial_sft_identity.json',
                'runlogs/development_preflight/preflight.json']
    for folder in ['eval/vlnce', 'vlnce_server/VLN_CE', 'tools/vllm_compat']:
        sources += [str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.yaml')]
    data = ROOT / 'data/datasets/R2R_VLNCE_v1-3_preprocessed/train'
    if sha(data / 'train.json.gz') != '340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea':
        raise ValueError('actual dataset differs from frozen source')
    sources += [str((data / name).relative_to(ROOT)) for name in ('train.json.gz', 'train_gt.json.gz')]
    evidence = list((SFTROOT / 'positive_initial_sft').glob('shard_*/log/stats_*_0.json'))
    if len(evidence) != 256:
        raise ValueError('SFT raw file count differs')
    evidence += list((SFTROOT / 'positive_initial_sft').glob('shard_*/summary.json'))
    if len(evidence) != 260:
        raise ValueError('SFT four-shard summary coverage differs')
    evidence += [SFTROOT / name for name in ('positive_initial_sft.completed',
                 'positive_initial_sft.validated.json', 'positive_initial_sft.dtype.json',
                 'vllm_positive_initial_sft.log')]
    evidence += [OLD / 'prepared_data/positive_initial_sft_identity.json']
    x = {'schema': 'three_direction_development_eval_v1',
         'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'training_identity_sha256': sha(parent), 'episodes': 256, 'scenes': 8,
         'group_size': 4, 'configured_training_seed': 11, 'total_updates_including_smoke': 64,
         'decode_seed': 11, 'dtype': 'float16', 'temperature': .2, 'top_p': .8,
         'max_completion_tokens': 512, 'max_turns': 12, 'max_pixels': 76800,
         'source_sha256': {rel: sha(ROOT / rel) for rel in sorted(set(sources))},
         'sft_reference': {'root': str(SFTROOT), 'label': 'positive_initial_sft',
             'checkpoint': json.loads((ROOT / 'prepared_data/positive_initial_sft_identity.json').read_text())['checkpoint'],
             'evidence_sha256': {str(p): sha(p) for p in sorted(evidence)},
             'full_weight_identity_checked_at_preflight': True,
             'scope': 'Reuse one completed FP16 decode reference, no independent SFT training seeds.'},
         'gate': {'each_metric_pp_at_least': 2.0,
                  'all_methods_require_vs_sft': True,
                  'turn_rloo_srgpo_additionally_require_vs_grpo': True},
         'scheduling': 'Wait for all three 64-step audits; two models on GPU0/1, each four GPU2 shards; GPU3 untouched.',
         'limits': ['Train-scene development reused adaptively; exploratory.',
                    'Configured rollout streams are not established as independent.',
                    'Simulator progress is privileged training-only feedback.',
                    'SRGPO-style process grouping and population outcome normalization are a composite.',
                    'No reserved or val-unseen inference in this pilot.']}
    out = ROOT / 'runlogs/freeze/evaluation_identity.json'
    with out.open('x') as f:
        f.write(json.dumps(x, indent=2) + '\n')
    print(json.dumps({'evaluation_identity_sha256': sha(out),
                      'source_files': len(x['source_sha256']), 'sft_evidence_files': len(evidence)}))


if __name__ == '__main__':
    main()
