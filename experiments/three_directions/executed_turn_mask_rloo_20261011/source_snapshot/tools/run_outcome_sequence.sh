#!/usr/bin/env bash
set -Eeuo pipefail
identity_sha=${1:?frozen identity sha}
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_executed_turn_rloo_20261011"
state="$root/runlogs/sequential_training"
mkdir -p "$state"
mkdir -p "$root/runlogs/development_suite"
exec 9>"$state/sequence.lock"
flock -n 9 || { echo "sequence already active" >&2; exit 2; }
on_exit() { rc=$?; if test "$rc" -ne 0; then printf '%s\n' "$rc" >"$state/sequence.failed"; fi; }
trap on_exit EXIT
rm -f "$state/sequence.failed"
cd "$root"
"$base/activevln_server_env/bin/python" tools/verify_outcome_freeze.py --root "$root" --identity-sha "$identity_sha" >"$state/freeze_before.json"
bash tools/prepare_outcome_refs.sh >"$root/runlogs/development_suite/prepare_refs.log" 2>&1
bash tools/start_outcome_service.sh >"$root/runlogs/service/start.log" 2>&1
bash tools/train_outcome_rloo.sh 2 0,1 >"$root/runlogs/td_executed_turn_rloo_n4_seed11/train_smoke.launcher.log" 2>&1
CUDA_VISIBLE_DEVICES="" "$base/activevln_train_env/bin/python" tools/audit_outcome_training.py --root "$root" --updates 2 --output "$root/runlogs/td_executed_turn_rloo_n4_seed11/smoke_audit.json" >"$root/runlogs/td_executed_turn_rloo_n4_seed11/smoke_audit.log"
"$base/activevln_server_env/bin/python" tools/verify_outcome_freeze.py --root "$root" --identity-sha "$identity_sha" >"$state/freeze_before_resume.json"
bash tools/train_outcome_rloo.sh 64 0,1 >"$root/runlogs/td_executed_turn_rloo_n4_seed11/train64.launcher.log" 2>&1
CUDA_VISIBLE_DEVICES="" "$base/activevln_train_env/bin/python" tools/audit_outcome_training.py --root "$root" --updates 64 --output "$root/runlogs/development_suite/outcome_consistent_rloo_train_audit.json" >"$root/runlogs/development_suite/outcome_consistent_rloo_train_audit.log"
pid=$(cat "$root/runlogs/service/server.pid")
ps -o args= -p "$pid" | grep -F "server.port=5087" >/dev/null
kill "$pid"
for _ in $(seq 1 60); do
  if ! curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1; then break; fi
  sleep 2
done
if curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1; then echo "owned Habitat service did not stop" >&2; exit 1; fi
for _ in $(seq 1 180); do
  m0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  m1=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
  m2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
  if test "$m0" -lt 10000 && test "$m1" -lt 10000 && test "$m2" -lt 10000; then break; fi
  sleep 10
done
test "$m0" -lt 10000 && test "$m1" -lt 10000 && test "$m2" -lt 10000
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/training.completed"
bash tools/evaluate_outcome_candidate.sh "$identity_sha" >"$root/runlogs/development_suite/evaluation.launcher.log" 2>&1
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/sequence.completed"
