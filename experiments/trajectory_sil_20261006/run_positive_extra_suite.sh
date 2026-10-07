#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
scale="$root/runlogs/positive_scale"
state="$root/runlogs/positive_extra"
mkdir -p "$state"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'extra evaluation already active' >&2; exit 2; }
if test -f "$state/suite.completed"; then exit 0; fi
test ! -f "$state/suite.failed"
worker_pids=()
cleanup() {
  status=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi
}
trap cleanup EXIT
# A lock file alone is not evidence of live training. Require its actual PID.
while ! test -f "$scale/suite.completed"; do
  test ! -f "$scale/suite.failed"
  test ! -f "$scale/scale.skipped"
  test -s "$scale/watcher.launcher.pid"
  kill -0 "$(cat "$scale/watcher.launcher.pid")"
  sleep 30
done
exec 8>"$scale/suite.lock"
flock 8
test ! -f "$scale/suite.failed"
test -f "$scale/training.completed"
test -f "$scale/three_seed_reserved_report.json"
cd "$root"
"$base/activevln_server_env/bin/python" - "$root" <<'PY'
import hashlib, json, sys
from pathlib import Path
r = Path(sys.argv[1])
protocol = json.loads((r / 'prepared_data/positive_extra_protocol_identity.json').read_text())
assert protocol['schema'] == 'positive_extra_evaluation_freeze_v1'
assert protocol['reserved_opened_at_freeze'] is False
for name, digest in protocol['source_sha256'].items():
    assert hashlib.sha256((r / name).read_bytes()).hexdigest() == digest, name
for seed in (11, 22, 33):
    for arm in ('control', 'candidate'):
        a = json.loads((r / f'runlogs/positive_scale/{arm}_seed{seed}_train_audit.json').read_text())
        assert a['expected_steps'] == a['observed_steps'] == 128 and a['arm'] == arm
        assert a['nonzero_actor_gradient_steps'] > 0 and a['positive_advantage_steps'] > 0
        assert a['kl_loss_metric_present_and_finite'] is True and a['max_terminal_score'] <= 20.001
        if arm == 'candidate': assert a['min_advantage'] >= -0.001
print('Frozen extra sources and six completed training audits verified.')
PY
wait_gpus() {
  for _ in $(seq 1 360); do
    busy=0
    for gpu in 0 1 2; do
      memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
      if test "$memory" -ge 10000; then busy=1; fi
    done
    if test "$busy" -eq 0; then return; fi
    sleep 10
  done
  echo 'GPUs remain occupied; no other process will be stopped' >&2; return 1
}
run_pair() {
  wait_gpus
  bash tools/run_positive_extra_model.sh "$1" "$2" "$3" 0 >"$state/$4.launcher.log" 2>&1 &
  first_pid=$!
  bash tools/run_positive_extra_model.sh "$5" "$6" "$7" 1 >"$state/$8.launcher.log" 2>&1 &
  second_pid=$!
  worker_pids=("$first_pid" "$second_pid")
  status=0
  wait "$first_pid" || status=1
  wait "$second_pid" || status=1
  worker_pids=()
  test "$status" -eq 0
}
run_pair development sft 11 sft_development reserved sft 11 sft_reserved
for role in development reserved; do
  if test "$role" = development; then
    trained="$root/runlogs/positive_development256"
  else
    trained="$root/runlogs/positive_reserved256"
  fi
  "$base/activevln_server_env/bin/python" tools/analyze_positive_sft_reference.py \
    --role "$role" --manifest "$root/prepared_data/${role}256.json" \
    --sft-root "$root/runlogs/positive_extra_${role}" --trained-root "$trained" \
    --compact "$state/${role}_with_sft.jsonl" --output "$state/${role}_with_sft.json" \
    >"$state/${role}_sft_analyze.log"
done
# Every fixed checkpoint is evaluated irrespective of its reserved result.
for seed in 11 22 33; do
  run_pair val_unseen control "$seed" "full_control_seed${seed}" \
    val_unseen candidate "$seed" "full_candidate_seed${seed}"
done
"$base/activevln_server_env/bin/python" tools/verify_three_seed_scale_raw.py \
  --candidate-root "$root/runlogs/positive_extra_val_unseen" \
  --control-root "$root/runlogs/positive_extra_val_unseen" \
  --candidate-pattern 'positive_trajectory_candidate_128step_seed{seed}' \
  --control-pattern 'positive_trajectory_control_128step_seed{seed}' \
  --manifest "$root/prepared_data/positive_full1839.json" \
  --compact-dir "$state" --output "$state/independent_three_seed_full_recount.json" \
  >"$state/full_three_seed_recount.log"
wait_gpus
bash tools/run_positive_extra_model.sh val_unseen sft 11 0 >"$state/sft_full.launcher.log" 2>&1
"$base/activevln_server_env/bin/python" tools/analyze_positive_sft_reference.py \
  --role val_unseen --manifest "$root/prepared_data/positive_full1839.json" \
  --sft-root "$root/runlogs/positive_extra_val_unseen" \
  --trained-root "$root/runlogs/positive_extra_val_unseen" \
  --compact "$state/full_with_sft.jsonl" --output "$state/full_with_sft.json" \
  >"$state/full_sft_analyze.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
