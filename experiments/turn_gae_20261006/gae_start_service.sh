#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_turn_gae_20261006"
run="$root/runlogs/service"
port=5060
test -f "$base/ActiveVLN_norm_terminal_rloo_20261006/runlogs/conditional_chain/chain.completed"
test -f "$root/runlogs/conditional_chain/normalized_failed_gate.completed"
mkdir -p "$run"
exec 9>"$run/start.lock"
flock -n 9 || { echo 'GAE service startup already active' >&2; exit 2; }
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >"$run/health.json" 2>/dev/null; then
  test -s "$run/server.pid" && kill -0 "$(cat "$run/server.pid")"
  exit 0
fi
for busy in 5057 5058 5059; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$busy/health" >/dev/null 2>&1; then
    echo "prior Habitat service is still active on $busy" >&2
    exit 1
  fi
done
used=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)
test "$used" -lt 10000 || { echo "GPU2 still busy: $used MiB" >&2; exit 1; }
cd "$root"
nohup env PYTHONUNBUFFERED=1 VLN_ORACLE_TURNWISE=0 \
  PYTHONPATH="$root/vlnce_server:$root${PYTHONPATH:+:$PYTHONPATH}" \
  RAY_TMPDIR=/dev/shm/td_srv_turn_gae_5060 \
  "$base/activevln_server_env/bin/python" -m vlnce_server.server \
  server.port="$port" 'vlnce.gpus=[2]' \
  'vlnce.r2r_gpu_plan=[16]' 'vlnce.rxr_gpu_plan=[0]' \
  >"$run/server.log" 2>&1 </dev/null 9>&- &
echo $! >"$run/server.pid"
for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >"$run/health.json" 2>/dev/null; then
    echo "GAE Habitat service healthy on $port"
    exit 0
  fi
  if ! kill -0 "$(cat "$run/server.pid")" 2>/dev/null; then
    tail -30 "$run/server.log" >&2
    exit 1
  fi
  sleep 2
done
echo 'GAE Habitat service did not become healthy' >&2
exit 1
