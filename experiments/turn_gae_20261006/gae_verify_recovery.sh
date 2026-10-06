#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_gae_20261006"
result="$root/runlogs/turn_gae_val_seen778"
python="$base/activevln_train_env/bin/python"
mkdir -p "$result"
exec 9>"$result/recovery.lock"
flock -n 9 || { echo 'GAE verifier recovery already active' >&2; exit 2; }
test -f "$result/suite.failed"
test ! -f "$result/suite.completed"
test ! -f "$result/suite.verified"
test ! -f "$result/verified_gate.json"
test "$(cat "$result/suite.failed")" = 1
grep -F "TypeError: 'type' object is not subscriptable" "$result/suite.launcher.log" >/dev/null
pid=$(cat "$result/suite.launcher.pid")
! kill -0 "$pid" 2>/dev/null || { echo 'original suite is still active' >&2; exit 1; }
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = 03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed
test "$(sha256sum "$root/tools/preverify_gae_suite.py" | awk '{print $1}')" = 0fc82ab795cf4de43715e6b3ce07639c5707a866d73fa02fba82a2cc003551bc
test "$(sha256sum "$root/tools/verify_gae_raw.py" | awk '{print $1}')" = 968461039bf760606d2051a3a378746aa4682376ca2804ff48fc26910e140319
"$python" - <<'PY'
import sys
assert sys.version_info >= (3, 9), sys.version
PY
for label in qwen3_exact_control_64step_seed11 turn_gae_64step_seed11; do
  test -f "$result/$label.completed"
  test ! -f "$result/$label.failed"
  "$python" - "$result/$label.validated.json" "$label" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['label']==sys.argv[2] and x['episodes']==778 and x['inference_errors']==0
PY
done
archive="$result/suite.failed.python38_preverify"
test ! -e "$archive"
mv "$result/suite.failed" "$archive"
on_exit() {
  status=$?
  if test "$status" -ne 0; then
    rm -f "$result/suite.completed" "$result/suite.verified" "$result/verified_gate.json"
    cp "$archive" "$result/suite.failed"
  fi
}
trap on_exit EXIT
"$python" "$root/tools/preverify_gae_suite.py" \
  "$result" "$result/manifest.json" "$result/precompletion_recount.json" \
  >"$result/precompletion_recount.recovery.log"
test -s "$result/precompletion_recount.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.completed"
"$python" "$root/tools/verify_gae_raw.py" \
  "$result" "$result/manifest.json" \
  --compact "$result/compact.json" \
  --output "$result/independent_recount.json" \
  >"$result/independent_recount.recovery.log"
"$python" - "$result" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
pre=json.loads((root/'precompletion_recount.json').read_text())
post=json.loads((root/'independent_recount.json').read_text())
assert pre['manifest_sha256']==post['manifest_sha256']=='03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed'
assert pre['episodes']==post['episodes']==778 and pre['scenes']==post['scenes']==53
assert pre['inference_errors']==post['inference_errors']==0
for key in ('control_successes','candidate_successes','candidate_only_success','control_only_success'):
    assert pre[key]==post[key],key
for key in ('control_spl','candidate_spl','paired_sr_points','paired_spl_points'):
    assert abs(pre[key]-post[key])<1e-6,key
assert pre['advance_gate_passed']==post['advance_gate_passed']
(root/'verified_gate.json').write_text(json.dumps({'advance_gate_passed':post['advance_gate_passed'],
  'paired_sr_points':post['paired_sr_points'],'paired_spl_points':post['paired_spl_points'],
  'precompletion_raw_stats_sha256':pre['raw_stats_sha256']},indent=2)+'\n')
PY
"$python" - "$result" <<'PY'
import json,sys,hashlib,datetime
from pathlib import Path
root=Path(sys.argv[1])
x={'schema':'gae_verifier_interpreter_recovery_v1',
   'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'original_failure':'Python 3.8.12 cannot evaluate tuple[int, float] annotation in preverify_gae_suite.py; both evaluator arms had already completed and validated 778 episodes with zero inference errors.',
   'original_failure_marker':'suite.failed.python38_preverify',
   'recovery_python':'activevln_train_env/bin/python 3.10.21',
   'raw_inference_rerun':False,
   'preverify_sha256':hashlib.sha256((root/'precompletion_recount.json').read_bytes()).hexdigest(),
   'independent_recount_sha256':hashlib.sha256((root/'independent_recount.json').read_bytes()).hexdigest()}
(root/'verification_recovery.json').write_text(json.dumps(x,indent=2)+'\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/suite.verified"
