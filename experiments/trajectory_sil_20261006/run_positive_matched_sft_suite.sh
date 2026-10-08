#!/usr/bin/env bash
set -euo pipefail
freeze_sha=${1:?frozen precision identity SHA256}
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
state="$root/runlogs/positive_matched_precision"
mkdir -p "$state"
exec 9>"$state/suite.lock"
flock -n 9 || { echo 'matched precision reference already active' >&2; exit 2; }
if test -f "$state/suite.completed"; then exit 0; fi
test ! -f "$state/suite.failed"
worker_pids=()
cleanup() {
  status=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi
}
trap cleanup EXIT
cd "$root"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" \
  tools/verify_positive_precision_freeze.py --root "$root" --identity-sha "$freeze_sha"
# Require actual owner identity while waiting. Completed parents may exit normally.
for parent in positive_scale positive_extra; do
  while ! test -f "$root/runlogs/$parent/suite.completed"; do
    test ! -f "$root/runlogs/$parent/suite.failed"
    CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" - "$root" "$parent" <<'PY'
from pathlib import Path
import sys
r, name = Path(sys.argv[1]), sys.argv[2]
pid = int((r / f'runlogs/{name}/watcher.launcher.pid').read_text())
process = Path(f'/proc/{pid}')
cmd = process.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
expected = {'positive_scale': 'tools/run_positive_scale_if_pass.sh',
            'positive_extra': 'tools/run_positive_extra_suite.sh'}[name]
assert expected in cmd and process.joinpath('cwd').resolve() == r
PY
    sleep 30
  done
done
exec 8>"$root/runlogs/positive_extra/suite.lock"
flock 8
test ! -f "$root/runlogs/positive_scale/suite.failed"
test ! -f "$root/runlogs/positive_extra/suite.failed"
cd "$root"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" \
  tools/verify_positive_precision_freeze.py --root "$root" --identity-sha "$freeze_sha"
# Confirm effective precision of every old comparator before adding the shared SFT reference.
for role in development reserved val_unseen; do
  if test "$role" = development; then
    trained="$root/runlogs/positive_development256"; steps=64; seeds=(11)
  elif test "$role" = reserved; then
    trained="$root/runlogs/positive_reserved256"; steps=128; seeds=(11 22 33)
  else
    trained="$root/runlogs/positive_extra_val_unseen"; steps=128; seeds=(11 22 33)
  fi
  for seed in "${seeds[@]}"; do
    for arm in control candidate; do
      label="positive_trajectory_${arm}_${steps}step_seed${seed}"
      CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py \
        --log "$trained/vllm_${label}.log" \
        --checkpoint "$root/verl_checkpoints/$label/global_step_$steps/actor/huggingface" \
        --expected-dtype float16 --output "$state/${role}_${label}.dtype.json"
    done
  done
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_engine_dtype.py \
    --log "$root/runlogs/positive_extra_${role}/vllm_positive_initial_sft.log" \
    --checkpoint "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" \
    --expected-dtype bfloat16 --output "$state/${role}_native_sft.dtype.json"
done
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
  echo 'GPUs remain occupied; no unrelated process will be stopped' >&2; return 1
}
wait_gpus
bash tools/run_positive_matched_sft_model.sh development 0 "$freeze_sha" >"$state/development.launcher.log" 2>&1 &
first_pid=$!
bash tools/run_positive_matched_sft_model.sh reserved 1 "$freeze_sha" >"$state/reserved.launcher.log" 2>&1 &
second_pid=$!
worker_pids=("$first_pid" "$second_pid")
status=0
wait "$first_pid" || status=1
wait "$second_pid" || status=1
worker_pids=()
test "$status" -eq 0
wait_gpus
bash tools/run_positive_matched_sft_model.sh val_unseen 0 "$freeze_sha" >"$state/val_unseen.launcher.log" 2>&1
for role in development reserved val_unseen; do
  if test "$role" = development; then
    trained="$root/runlogs/positive_development256"; manifest="$root/prepared_data/development256.json"
  elif test "$role" = reserved; then
    trained="$root/runlogs/positive_reserved256"; manifest="$root/prepared_data/reserved256.json"
  else
    trained="$root/runlogs/positive_extra_val_unseen"; manifest="$root/prepared_data/positive_full1839.json"
  fi
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/analyze_positive_sft_reference.py \
    --role "$role" --manifest "$manifest" \
    --sft-root "$root/runlogs/positive_matched_precision_${role}" --trained-root "$trained" \
    --compact "$state/${role}_with_sft.jsonl" --output "$state/${role}_with_sft.json" \
    >"$state/${role}_analyze.log"
  args=()
  if test "$role" = val_unseen; then
    args=(--full-three-seed-report "$root/runlogs/positive_extra/independent_three_seed_full_recount.json")
  fi
  CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" tools/verify_positive_sft_compact.py \
    --role "$role" --manifest "$manifest" --compact "$state/${role}_with_sft.jsonl" \
    --report "$state/${role}_with_sft.json" \
    --sft-validators "$root/runlogs/positive_matched_precision_${role}" \
    --trained-validators "$trained" --freeze "$root/prepared_data/positive_extra_protocol_identity.json" \
    --sft-identity "$root/prepared_data/positive_initial_sft_identity.json" \
    --output "$state/${role}_independent_recount.json" "${args[@]}" \
    >"$state/${role}_independent_recount.log"
done
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" \
  tools/verify_positive_precision_freeze.py --root "$root" --identity-sha "$freeze_sha"
CUDA_VISIBLE_DEVICES="" "$base/activevln_server_env/bin/python" - "$root" "$freeze_sha" <<'PY'
from pathlib import Path
import hashlib, json, sys
r, digest = Path(sys.argv[1]), sys.argv[2]
state = r / 'runlogs/positive_matched_precision'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
identity = r / 'prepared_data/positive_matched_precision_identity.json'
assert sha(identity) == digest
out = {'schema': 'positive_matched_precision_comparison_identity_v1',
       'precision_freeze_sha256': digest, 'actual_matched_sft_engine_dtype': 'float16',
       'actual_trained_engine_dtype': 'float16', 'native_sft_engine_dtype': 'bfloat16',
       'sft_training_seeds': 0, 'sft_decode_seed': 11, 'screens': {},
       'interpretation': 'Additional shared FP16 SFT reference, frozen before SFT results; reuses every planned trained raw evaluation. Exploratory configured-seed comparisons, not independent SFT training replications.'}
for role in ('development', 'reserved', 'val_unseen'):
    sft = r / f'runlogs/positive_matched_precision_{role}'
    proof = json.loads((sft / 'positive_initial_sft.dtype.json').read_text())
    assert proof['actual_engine_dtype'] == 'float16'
    assert sha(sft / 'vllm_positive_initial_sft.log') == proof['log_snapshot_sha256']
    files = [state / f'{role}_with_sft.json', state / f'{role}_with_sft.jsonl',
             state / f'{role}_independent_recount.json',
             sft / 'positive_initial_sft.validated.json', sft / 'positive_initial_sft.dtype.json']
    files += list(state.glob(f'{role}_*.dtype.json'))
    out['screens'][role] = {'sft_result_root': str(sft),
                           'files_sha256': {str(p.relative_to(r)): sha(p) for p in files}}
with (state / 'verified_precision_comparisons.json').open('x') as handle:
    handle.write(json.dumps(out, indent=2) + '\n')
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"
