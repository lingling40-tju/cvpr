#!/usr/bin/env bash
set -euo pipefail
role=${1:?development, reserved, or val_unseen}
arm=${2:?sft, control, or candidate}
seed=${3:?11, 22, or 33}
gpu=${4:?0 or 1}
case "$gpu" in 0) port=8139 ;; 1) port=8140 ;; *) exit 2 ;; esac
case "$seed" in 11|22|33) ;; *) exit 2 ;; esac
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_positive_trajectory_20261006"
state="$root/runlogs/positive_extra"
case "$role" in
  development) manifest="$root/prepared_data/development256.json" ;;
  reserved) manifest="$root/prepared_data/reserved256.json" ;;
  val_unseen) manifest="$root/prepared_data/positive_full1839.json" ;;
  *) exit 2 ;;
esac
case "$arm" in
  sft)
    test "$seed" = 11
    label=positive_initial_sft
    checkpoint="$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn"
    ;;
  control|candidate)
    test "$role" = val_unseen
    label="positive_trajectory_${arm}_128step_seed${seed}"
    checkpoint="$root/verl_checkpoints/$label/global_step_128/actor/huggingface"
    test -f "$root/runlogs/$label/completed"
    ;;
  *) exit 2 ;;
esac
result="$root/runlogs/positive_extra_${role}"
# All extra inference waits for the original six-model reserved procedure.
test -f "$root/runlogs/positive_scale/suite.completed"
test ! -f "$root/runlogs/positive_scale/suite.failed"
test -f "$checkpoint/config.json"
mkdir -p "$result" "$root/runlogs/gpu_eval_locks"
exec 8>"$root/runlogs/gpu_eval_locks/gpu${gpu}.lock"
flock 8
exec 9>"$result/$label.lock"
flock 9
cd "$root"
if test -f "$result/$label.completed"; then
  test ! -f "$result/$label.failed"
  "$base/activevln_server_env/bin/python" tools/validate_positive_extra.py \
    --root "$result" --manifest "$manifest" --role "$role" --label "$label" \
    --output "$result/$label.validated.json"
  exit 0
fi
test ! -f "$result/$label.failed"
if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
  echo "port $port is occupied" >&2; exit 1
fi
for device in "$gpu" 2; do
  memory=$(nvidia-smi -i "$device" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$memory" -lt 10000
done
if curl -fsS --max-time 1 http://127.0.0.1:5085/health >/dev/null 2>&1; then
  echo 'Training Habitat service still running' >&2; exit 1
fi
if test "$arm" = sft; then
  "$base/activevln_server_env/bin/python" - "$root/prepared_data/positive_initial_sft_identity.json" "$checkpoint" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); c = Path(sys.argv[2])
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'ea2a5cb50422f1ef3967215d3b55ac9fffe7e233d111d51693a2b7781a44fe47'
m = json.loads(p.read_text())
assert m['checkpoint'] == str(c) and m['rl_optimizer_steps'] == 0
assert {x.name for x in c.iterdir() if x.is_file() and not x.name.startswith('.')} == set(m['files'])
for name, identity in m['files'].items():
    f = c / name; h = hashlib.sha256()
    assert f.stat().st_size == identity['bytes']
    with f.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''): h.update(chunk)
    assert h.hexdigest() == identity['sha256']
print('Unchanged SFT file identity verified; no optimizer updates.')
PY
fi
export PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/tools/vllm_compat:$PYTHONPATH" \
  "$base/activevln_train_env/bin/vllm" serve "$checkpoint" --port "$port" \
  --max-model-len 16384 --gpu-memory-utilization 0.72 --trust-remote-code \
  --disable-log-requests --seed 11 >"$result/vllm_${label}.log" 2>&1 &
server_pid=$!
worker_pids=()
cleanup() {
  status=$?
  for pid in "${worker_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  if test "$status" -ne 0; then printf '%s\n' "$status" >"$result/$label.failed"; fi
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 3
done
test "$ready" -eq 1
for shard in 0 1 2 3; do
  if test "$role" = val_unseen; then
    eval_args=(tools/eval_positive_full_unseen.py)
  else
    eval_args=(tools/eval_train_scene_subset.py --role "$role" --count 256)
  fi
  CUDA_VISIBLE_DEVICES=2 "$base/activevln_server_env/bin/python" "${eval_args[@]}" \
    --model-label "$label" --manifest "$manifest" --result-root "$result" \
    --shard-count 4 --shard-index "$shard" --max-turns 12 \
    --base-url "http://127.0.0.1:$port/v1" \
    >"$result/eval_${label}_shard${shard}.log" 2>&1 &
  worker_pids+=("$!")
done
status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
"$base/activevln_server_env/bin/python" tools/validate_positive_extra.py \
  --root "$result" --manifest "$manifest" --role "$role" --label "$label" \
  --output "$result/$label.validated.json" >"$result/$label.validation.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$result/$label.completed"
