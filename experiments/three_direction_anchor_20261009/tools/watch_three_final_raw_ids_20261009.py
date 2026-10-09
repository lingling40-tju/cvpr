"""CPU-only handoff to the supplemental verifier after original suite success."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

EVALUATION_SHA = '807a815a3de2143215f8e8075a086c9d02a89b5ca14a7c978191ee308cd2d93e'
SUPPLEMENT_SHA = '9260ebbb393912269268e878297e390a0eafe3d62d3bf45457a522417ad93d39'
VERIFIER_SHA = '547721f41b38298bfb83804bd6e8c7628d4ad0e806cbbcb42dac431fabb86908'
BASE = Path('/Knowin/foundation/haozhiwang/whz')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def publish_status(state, value):
    temporary = state / ('status.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(dict(value, utc=utc()), indent=2) + '\n')
    temporary.replace(state / 'status.json')


def check_sources(root):
    assert sha(root / 'runlogs/freeze/evaluation_identity.json') == EVALUATION_SHA
    assert sha(root / 'runlogs/freeze/raw_id_supplement_identity.json') == SUPPLEMENT_SHA
    assert sha(root / 'tools/verify_three_development_raw_ids_20261009.py') == VERIFIER_SHA
    supplement = json.loads((root / 'runlogs/freeze/raw_id_supplement_identity.json').read_text())
    assert supplement['parent_evaluation_identity_sha256'] == EVALUATION_SHA
    assert supplement['development_opened_at_supplement_freeze'] is False


def check_owner(root):
    suite = root / 'runlogs/development_suite'
    launch = json.loads((suite / 'launch.json').read_text())
    pid = int((suite / 'suite.launcher.pid').read_text().strip())
    assert pid == launch['pid'] and launch['evaluation_identity_sha256'] == EVALUATION_SHA
    proc = Path('/proc') / str(pid)
    command = [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v]
    assert len(command) == 3 and Path(command[0]).name == 'bash'
    assert command[1:] == ['tools/run_three_dev_suite.sh', EVALUATION_SHA]
    assert Path(os.readlink(str(proc / 'cwd'))).resolve() == root
    return {'pid': pid, 'command': command, 'cwd': str(root)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    suite = root / 'runlogs/development_suite'
    output = suite / 'final_raw_id_independent_recount.json'
    check_sources(root)
    assert not (suite / 'suite.failed').exists()
    assert not output.exists(), 'Final report already exists; inspect it before starting another handoff.'
    owner = check_owner(root) if not (suite / 'suite.completed').exists() else None
    if args.preflight:
        print(json.dumps({'status': 'PASS', 'owner': owner, 'output_absent': True,
                          'scope': 'CPU source and live owner checks; no evaluator or model call.'}))
        return
    state = root / 'runlogs/final_raw_id_verification'
    state.mkdir(exist_ok=True)
    with (state / 'watcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Final raw-ID handoff already active.', file=sys.stderr)
            return 2
        try:
            launch = {'created_utc': utc(), 'pid': os.getpid(), 'owner': owner,
                      'watcher_sha256': sha(Path(__file__)), 'verifier_sha256': VERIFIER_SHA,
                      'evaluation_identity_sha256': EVALUATION_SHA, 'supplement_identity_sha256': SUPPLEMENT_SHA,
                      'output': str(output), 'scope': 'CPU only; original sources, gates and markers unchanged.'}
            with (state / 'launch.json').open('x') as stream:
                stream.write(json.dumps(launch, indent=2) + '\n')
            while not (suite / 'suite.completed').exists():
                assert not (suite / 'suite.failed').exists(), 'Original evaluation suite failed.'
                try:
                    owner = check_owner(root)
                except (OSError, AssertionError):
                    time.sleep(2)
                    if (suite / 'suite.completed').exists():
                        break
                    raise
                publish_status(state, {'state': 'waiting_for_original_suite', 'owner': owner})
                time.sleep(30)
            assert not (suite / 'suite.failed').exists()
            check_sources(root)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            python = str(BASE / 'activevln_server_env/bin/python')
            publish_status(state, {'state': 'verifying_original_freeze_and_raw_ids'})
            with (state / 'freeze_before_final.log').open('x') as stream:
                subprocess.run([python, 'tools/verify_three_eval_freeze.py', '--root', str(root),
                                '--identity-sha', EVALUATION_SHA], cwd=str(root), env=environment,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            with (state / 'verifier.log').open('x') as stream:
                subprocess.run([python, 'tools/verify_three_development_raw_ids_20261009.py',
                                '--root', str(root), '--output', str(output)], cwd=str(root),
                               env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            report = json.loads(output.read_text())
            assert report['status'] == 'PASS' and report['raw_internal_ids_checked'] == 1024
            assert report['verifier_sha256'] == VERIFIER_SHA
            assert report['aggregate_sha256'] == sha(suite / 'three_direction_report.json')
            publish_status(state, {'state': 'verification_completed', 'report': str(output),
                                   'report_sha256': sha(output), 'raw_internal_ids_checked': 1024})
            with (state / 'watcher.completed').open('x') as stream:
                stream.write(utc() + '\n')
        except Exception:
            detail = traceback.format_exc()
            if not (state / 'failure.json').exists():
                with (state / 'failure.json').open('x') as stream:
                    stream.write(json.dumps({'utc': utc(), 'error': detail}, indent=2) + '\n')
            publish_status(state, {'state': 'handoff_failed', 'error': detail})
            raise


if __name__ == '__main__':
    sys.exit(main())
