#!/usr/bin/env bash
set -euo pipefail

root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_three_directions_20261002
run_dir="$root/runlogs/three_direction_pilot_suite"
mkdir -p "$run_dir"
exec 9>"$run_dir/suite.lock"
flock -n 9 || { echo 'suite already active' >&2; exit 2; }
cd "$root"

for mode in branch recovery counterfactual; do
  test -f "$root/runlogs/three_directions_${mode}_2step/completed" || {
    echo "smoke not complete: $mode" >&2
    exit 1
  }
done
curl -fsS --max-time 5 http://127.0.0.1:5002/health >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:5007/health >/dev/null
rm -f "$run_dir/suite.failed" "$run_dir/suite.completed"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.started"

nohup env VLN_PILOT_SERVICE_URL=http://127.0.0.1:5002 \
  bash tools/run_direction_pilot.sh branch 64 0,1 \
  >"$run_dir/branch.launcher.log" 2>&1 < /dev/null &
branch_pid=$!
echo "$branch_pid" >"$run_dir/branch.pid"
nohup env VLN_PILOT_SERVICE_URL=http://127.0.0.1:5007 \
  bash tools/run_direction_pilot.sh recovery 64 2,3 \
  >"$run_dir/recovery.launcher.log" 2>&1 < /dev/null &
recovery_pid=$!
echo "$recovery_pid" >"$run_dir/recovery.pid"

branch_status=0
wait "$branch_pid" || branch_status=$?
echo "$branch_status" >"$run_dir/branch.exit"

cf_status=0
if [ "$branch_status" -eq 0 ]; then
  curl -fsS --max-time 5 http://127.0.0.1:5002/health >/dev/null
  nohup env VLN_PILOT_SERVICE_URL=http://127.0.0.1:5002 \
    bash tools/run_direction_pilot.sh counterfactual 64 0,1 \
    >"$run_dir/counterfactual.launcher.log" 2>&1 < /dev/null &
  cf_pid=$!
  echo "$cf_pid" >"$run_dir/counterfactual.pid"
else
  echo 'branch failed; counterfactual requires manual service check' >&2
  cf_status=1
fi

recovery_status=0
wait "$recovery_pid" || recovery_status=$?
echo "$recovery_status" >"$run_dir/recovery.exit"
if [ "$cf_status" -eq 0 ]; then
  wait "$cf_pid" || cf_status=$?
fi
echo "$cf_status" >"$run_dir/counterfactual.exit"

if [ "$branch_status" -ne 0 ] || [ "$recovery_status" -ne 0 ] || [ "$cf_status" -ne 0 ]; then
  date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.failed"
  exit 1
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/suite.completed"
