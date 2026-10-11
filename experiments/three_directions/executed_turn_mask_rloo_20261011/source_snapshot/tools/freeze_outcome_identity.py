"""Freeze code, data, reference predictions, screen, and gates before training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/Knowin/foundation/haozhiwang/whz')
SOURCES = (
    'PILOT_PROTOCOL.md',
    'examples/vlnce/train_vlnce_4gpus.yaml',
    'eval/vlnce/eval_vlnce.py',
    'prepared_data/development256.json',
    'prepared_data/fit512.parquet',
    'prepared_data/fit512_manifest.json',
    'tools/aggregate_outcome_report.py',
    'tools/analyze_train_scene_pair.py',
    'tools/audit_outcome_training.py',
    'tools/eval_train_scene_subset.py',
    'tools/freeze_outcome_identity.py',
    'tools/prepare_outcome_refs.sh',
    'tools/start_outcome_service.sh',
    'tools/train_outcome_rloo.sh',
    'tools/test_outcome_credit.py',
    'tools/test_executed_turn_mask.py',
    'tools/run_outcome_sequence.sh',
    'tools/evaluate_outcome_candidate.sh',
    'tools/validate_train_label.py',
    'tools/verify_outcome_freeze.py',
    'tools/verify_outcome_raw_ids.py',
    'tools/verify_positive_compact.py',
    'tools/verify_positive_engine_dtype.py',
    'verl/trainer/ppo/ray_trainer.py',
    'verl/trainer/ppo/turn_rloo_advantage.py',
    'verl/workers/agent/parallel_env_vlnce.py',
    'vlnce_server/env.py',
    'vlnce_server/outcome_credit.py',
    'vlnce_server/semantic_reward/env.py',
)
OLD = BASE / 'ActiveVLN_three_direction_20261009/runlogs/development256'
SFT = BASE / 'ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision_development/positive_initial_sft'
SFT_MODEL_ID = BASE / 'ActiveVLN_positive_trajectory_20261006/prepared_data/positive_initial_sft_identity.json'
REFS = {
    'sft': {
        'raw_root': SFT,
        'files': (OLD / 'positive_initial_sft.completed', OLD / 'positive_initial_sft.validated.json',
                  OLD / 'positive_initial_sft.dtype.json', SFT_MODEL_ID),
    },
    'turn_rloo': {
        'raw_root': OLD / 'td_turn_rloo_n4_seed11',
        'files': (OLD / 'td_turn_rloo_n4_seed11.completed', OLD / 'td_turn_rloo_n4_seed11.validated.json',
                  OLD / 'td_turn_rloo_n4_seed11.dtype.json',
                  BASE / 'ActiveVLN_three_direction_20261009/runlogs/development_suite/turn_rloo_train_audit.json'),
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    state = ROOT / 'runlogs/freeze'
    state.mkdir(parents=True, exist_ok=True)
    out = state / 'identity.json'
    if out.exists() or (ROOT / 'runlogs/development_suite/development.opened').exists():
        raise FileExistsError('freeze already written or development inference already opened')
    source_sha = {rel: digest(ROOT / rel) for rel in SOURCES}
    reference_sha = {}
    reference_counts = {}
    for label, ref in REFS.items():
        raw_files = sorted(ref['raw_root'].glob('shard_*/log/stats_*_0.json'))
        if len(raw_files) != 256:
            raise ValueError(f'{label} reference does not have exactly 256 raw episodes')
        for path in ref['files']:
            if not path.is_file():
                raise FileNotFoundError(path)
        reference_sha[label] = {
            str(path): digest(path) for path in list(ref['files']) + raw_files
        }
        reference_counts[label] = {'raw_episode_records': len(raw_files)}
    payload = {
        'schema': 'executed_turn_mask_rloo_n4_development_freeze_v1',
        'remote_root': str(ROOT),
        'parent_three_direction_identity_sha256': digest(BASE / 'ActiveVLN_three_direction_20261009/runlogs/freeze/identity.json'),
        'candidate': 'outcome_consistent_rloo_n4_seed11',
        'method': 'same outcome-consistent turn-level RLOO; additionally excludes only the final generated turn explicitly marked forced_stop_reason=turn_budget with empty executed_actions from turn-return and policy-credit mapping',
        'budget': {'fit_rows': 512, 'optimizer_updates_including_smoke': 64, 'smoke_updates': 2,
                   'resume_updates': 62, 'batch_rows': 8, 'group_size': 4,
                   'configured_seed': 11, 'learning_rate': 1e-6, 'reference_kl': 0.1,
                   'progress_weight': 0.5, 'gamma': 1.0, 'success_reward': 15,
                   'successful_ndtw_weight': 5, 'max_turns': 12, 'max_steps': 36},
        'evaluation': {'manifest_sha256': digest(ROOT / 'prepared_data/development256.json'),
                       'episodes': 256, 'scenes': 8, 'dtype': 'float16', 'decode_seed': 11,
                       'temperature': 0.2, 'top_p': 0.8, 'max_tokens': 512,
                       'turns': 12, 'pixels': 76800, 'reserved_opened': False,
                       'val_unseen_opened': False},
        'gate': {'vs_sft_sr_points': 2.0, 'vs_sft_spl_points': 2.0,
                 'vs_previous_rloo_sr_points': 2.0, 'vs_previous_rloo_spl_points': 2.0},
        'references': reference_sha,
        'reference_counts': reference_counts,
        'source_sha256': source_sha,
        'limits': [
            'Single configured training seed; independent rollout-stream RNG not established.',
            'Development train scenes were adaptively reused in prior work; this is not a clean test.',
            'RLOO process signal is privileged simulator geodesic progress.',
            'This pilot changes only policy-credit masking for explicitly recorded nonexecuted terminal budget-overflow turns; rewards, optimizer settings, and gates match the prior outcome-consistent RLOO pilot.',
            'Positive result, if any, is a one-seed pilot and not deployable semantic reward evidence.'
        ],
    }
    payload['identity_sha256'] = canonical_sha(payload)
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'status': 'FROZEN', 'identity_sha256': payload['identity_sha256'],
                      'source_count': len(source_sha),
                      'reference_records': {k: v['raw_episode_records'] for k, v in reference_counts.items()}}))


if __name__ == '__main__':
    main()
