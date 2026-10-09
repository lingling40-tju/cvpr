"""CPU-only checks of reused real SFT evidence and evaluation/audit interfaces."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


BASE = Path('/Knowin/foundation/haozhiwang/whz')
ROOT = BASE / 'ActiveVLN_three_direction_20261009'
OLD = BASE / 'ActiveVLN_positive_trajectory_20261006'
SFT = OLD / 'runlogs/positive_matched_precision_development'
STATE = ROOT / 'runlogs/development_preflight'
TRAIN = str(BASE / 'activevln_train_env/bin/python')
SERVER = str(BASE / 'activevln_server_env/bin/python')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args, *, good=True):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=str(ROOT / 'vlnce_server') + ':' + str(ROOT))
    x = subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if (x.returncode == 0) != good:
        raise RuntimeError(str(args) + '\n' + x.stdout[-2000:] + x.stderr[-3000:])
    return x


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    identity_path = ROOT / 'prepared_data/positive_initial_sft_identity.json'
    identity = json.loads(identity_path.read_text())
    checkpoint = Path(identity['checkpoint'])
    checked = {}
    for rel, spec in identity['files'].items():
        path = checkpoint / rel
        if path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
            raise ValueError('SFT identity differs: ' + rel)
        checked[rel] = spec['sha256']
    matches = {}
    for rel in ['eval/vlnce/eval_vlnce.py', 'tools/eval_train_scene_subset.py',
                'tools/validate_train_label.py', 'tools/analyze_train_scene_pair.py',
                'tools/verify_positive_compact.py', 'tools/verify_positive_engine_dtype.py']:
        if sha(ROOT / rel) != sha(OLD / rel):
            raise ValueError('reused evaluator differs: ' + rel)
        matches[rel] = sha(ROOT / rel)
    for p in (ROOT / 'vlnce_server/VLN_CE').rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.yaml'):
            rel = p.relative_to(ROOT)
            if sha(p) != sha(OLD / rel):
                raise ValueError('Habitat source differs: ' + str(rel))
            matches[str(rel)] = sha(p)
    run([SERVER, 'tools/validate_train_label.py', '--root', str(SFT),
         '--manifest', 'prepared_data/development256.json', '--role', 'development',
         '--label', 'positive_initial_sft', '--output', str(STATE / 'sft_raw_validator.json')])
    validator = json.loads((STATE / 'sft_raw_validator.json').read_text())
    dev_manifest = json.loads((ROOT / 'prepared_data/development256.json').read_text())
    if validator['episodes'] != 256 or len(set(dev_manifest['scene_ids'])) != 8 or validator['inference_errors']:
        raise ValueError('SFT coverage differs')
    run([SERVER, 'tools/verify_positive_engine_dtype.py', '--log',
         str(SFT / 'vllm_positive_initial_sft.log'), '--checkpoint', str(checkpoint),
         '--expected-dtype', 'float16', '--output', str(STATE / 'sft_actual_dtype.json')])
    shards = []
    for shard in range(4):
        x = run([SERVER, 'tools/eval_train_scene_subset.py', '--model-label', 'cpu_validation',
                 '--manifest', 'prepared_data/development256.json', '--result-root', str(STATE),
                 '--role', 'development', '--count', '256', '--shard-count', '4',
                 '--shard-index', str(shard), '--validate-only'])
        rows = [json.loads(line) for line in x.stdout.splitlines() if line.startswith('{')]
        row = rows[-1]
        if row['shard_count'] != 64 or row['scene_count'] != 8 or row['missing_ndtw_references']:
            raise ValueError('CPU shard or nDTW coverage differs')
        shards.append(row)
    # A same-file alias tests the real raw->compact->independent recount chain.
    # No alias report is retained as a navigation result.
    with tempfile.TemporaryDirectory(prefix='td_sft_self_pair_') as temp:
        p = Path(temp)
        for label in ('self_a', 'self_b'):
            (p / label).symlink_to(SFT / 'positive_initial_sft', target_is_directory=True)
            (p / (label + '.completed')).write_text('CPU self-pair fixture\n')
            run([SERVER, 'tools/validate_train_label.py', '--root', str(p),
                 '--manifest', 'prepared_data/development256.json', '--role', 'development',
                 '--label', label, '--output', str(p / (label + '.validated.json'))])
        run([SERVER, 'tools/analyze_train_scene_pair.py', '--root', str(p),
             '--manifest', 'prepared_data/development256.json', '--role', 'development',
             '--control', 'self_a', '--candidate', 'self_b', '--compact', str(p / 'pair.jsonl'),
             '--output', str(p / 'pair.json')])
        run([SERVER, 'tools/verify_positive_compact.py', '--manifest',
             'prepared_data/development256.json', '--compact', str(p / 'pair.jsonl'),
             '--report', str(p / 'pair.json'), '--validators', str(p),
             '--output', str(p / 'independent.json')])
        recount = json.loads((p / 'independent.json').read_text())
        if recount['paired_sr_points'] != 0 or recount['paired_spl_points'] != 0:
            raise ValueError('same-file CPU recount must be zero')
    report = {'schema': 'three_direction_cpu_eval_preflight_v1', 'status': 'PASS',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model_calls': 0, 'gpu_tasks_started': 0, 'sft_identity_sha256': sha(identity_path),
              'sft_files_full_sha256_checked': checked, 'reused_source_sha256': matches,
              'sft_raw_validator': validator, 'shards': shards,
              'same_file_raw_to_compact_recount_passed': True,
              'scope': 'Reused real FP16 SFT evidence and CPU data/analysis interfaces; no new candidate navigation result.'}
    with (STATE / 'preflight.json').open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'sft_successes': validator['successes'],
                      'spl': validator['spl'], 'sources': len(matches), 'model_calls': 0}))


if __name__ == '__main__':
    main()
