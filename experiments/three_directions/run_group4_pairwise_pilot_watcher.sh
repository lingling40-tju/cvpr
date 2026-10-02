#!/usr/bin/env bash
set -euo pipefail

base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
root="$base/ActiveVLN_group4_pairwise_20261002"
run_dir="$root/runlogs/group4_pairwise_64pilot"
result="$source_root/runlogs/three_direction_val256"
mkdir -p "$run_dir"
exec 9>"$run_dir/watcher.lock"
flock -n 9 || { echo 'pairwise pilot watcher already active' >&2; exit 2; }
if test -f "$run_dir/watcher.completed"; then exit 0; fi
rm -f "$run_dir/watcher.failed"

stop_service() {
  pid_file="$root/runlogs/group4_pairwise_service/server.pid"
  if test -s "$pid_file"; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      ps -o args= -p "$pid" | grep -F 'server.port=5017' >/dev/null || return 1
      kill "$pid"
    fi
  fi
  for attempt in $(seq 1 30); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:5017/health >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  ! curl -fsS --max-time 2 http://127.0.0.1:5017/health >/dev/null 2>&1
}
on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then
    stop_service || true
    printf '%s\n' "$status" >"$run_dir/watcher.failed"
  fi
}
trap on_exit EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.started"
test -f "$root/runlogs/three_directions_group4_pairwise_2step_seed11/completed"
test -f "$source_root/runlogs/three_directions_group4_64step_seed11/completed"
test -f "$result/group4_64_seed11.completed"
test -f "$result/branch_control64.completed"
test "$(sha256sum "$result/manifest.json" | awk '{print $1}')" = \
  546581366e7030764ad1a16d7c43f9044906d02348a4ad00d79c89fe2fe5ac46

bash "$root/tools/start_group4_pairwise_service.sh" 1 >"$run_dir/service.log" 2>&1
VLN_PAIRWISE_GPUS=0,1 bash "$root/tools/run_group4_pairwise_training.sh" 64 11 \
  >"$run_dir/train.launcher.log" 2>&1
test -f "$root/runlogs/three_directions_group4_pairwise_64step_seed11/completed"
test -s "$root/runlogs/three_directions_group4_pairwise_64step_seed11/paired_train_audit.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/train.completed"

stop_service
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/service.stopped"
checkpoint="$root/verl_checkpoints/three_directions_group4_pairwise_64step_seed11/global_step_64/actor/huggingface"
test -f "$checkpoint/config.json"
VLN_EVAL_PORT=8018 bash "$source_root/tools/run_direction_eval.sh" \
  group4_pairwise64_seed11 "$checkpoint" 1 0 \
  >"$run_dir/eval.launcher.log" 2>&1
test -f "$result/group4_pairwise64_seed11.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/eval.completed"

for comparison in 'group4_64_seed11:group4' 'branch_control64:control'; do
  control=${comparison%%:*}
  short=${comparison#*:}
  analysis="$result/paired_group4_pairwise64_vs_${control}.json"
  "$base/activevln_server_env/bin/python" "$source_root/tools/analyze_matched_pair.py" \
    --root "$result" --candidate group4_pairwise64_seed11 --control "$control" \
    --expected-count 256 --output "$analysis" \
    >"$run_dir/analyze_${short}.log" 2>&1
  test -s "$analysis"
  "$base/activevln_server_env/bin/python" "$source_root/tools/package_val256_pair.py" \
    --result-root "$result" --candidate group4_pairwise64_seed11 \
    --control "$control" --analysis "$analysis" \
    --output-dir "$root/runlogs/pairwise64_package/${short}" \
    >"$run_dir/package_${short}.log" 2>&1
  test -s "$root/runlogs/pairwise64_package/${short}/package.json"
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/${short}.completed"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/watcher.completed"
