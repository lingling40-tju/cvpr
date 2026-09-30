#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_semantic_20260930"
log_dir="$root/runlogs"
mkdir -p "$log_dir"
cd "$root"

healthy() {
  curl --silent --fail --max-time 2 "http://127.0.0.1:$1/health" >/dev/null
}

wait_healthy() {
  local port=$1
  local name=$2
  for ((attempt=1; attempt<=90; attempt++)); do
    if healthy "$port"; then
      echo "$name ready on 127.0.0.1:$port"
      return 0
    fi
    sleep 2
  done
  echo "$name did not become healthy; see $log_dir" >&2
  return 1
}

if ! healthy 5003; then
  nohup env PYTHONUNBUFFERED=1 \
    "$base/activevln_semantic_verifier_env/bin/python" \
    tools/semantic_verifier_server.py \
    --model-path "$base/models/Qwen3-VL-8B-Instruct" --port 5003 \
    >"$log_dir/semantic_verifier_server_8b.log" 2>&1 </dev/null &
  echo $! >"$log_dir/semantic_verifier_server.pid"
  wait_healthy 5003 "semantic verifier"
fi

if ! healthy 5002; then
  nohup env PYTHONUNBUFFERED=1 \
    PYTHONPATH="$root/vlnce_server:$root:${PYTHONPATH:-}" \
    RAY_TMPDIR=/tmp/avse_srv_d \
    "$base/activevln_server_env/bin/python" \
    -m vlnce_server.server 'server.port=5002' 'vlnce.gpus=[3]' \
    'vlnce.r2r_gpu_plan=[8]' 'vlnce.rxr_gpu_plan=[0]' \
    >"$log_dir/semantic_vlnce_server.log" 2>&1 </dev/null &
  echo $! >"$log_dir/semantic_vlnce_server.pid"
  wait_healthy 5002 "VLN-CE simulator"
fi
