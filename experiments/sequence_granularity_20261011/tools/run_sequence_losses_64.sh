#!/usr/bin/env bash
set -Eeuo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_sequence_granularity_20261011"
state="$root/runlogs/sequence_training"
mkdir -p "$state"
exec 9>"$state/sequence.lock"
flock -n 9 || { echo "sequence already active" >&2; exit 2; }
on_exit() { rc=$?; if test "$rc" -ne 0; then printf "%s\n" "$rc" >"$state/sequence.failed"; fi; }
trap on_exit EXIT
rm -f "$state/sequence.failed"
cd "$root"
test -s "$state/freeze_identity.json"
test "$(sha256sum prepared_data/fit512.parquet | awk '{print $1}')" = 240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e
test "$(sha256sum prepared_data/development256.json | awk '{print $1}')" = 8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3
test ! -e "$state/training.completed"
for method in gspo_sequence gmpo_geomean dhpo_average; do
  run="$root/runlogs/seq_${method}_n4_seed11"
  test -f "$run/smoke.completed"
  test ! -f "$run/failed"
  test -s "$run/smoke_gradient_audit.json"
  printf "%s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ') starting 64-step continuation for $method" >>"$state/sequence.log"
  bash "$root/tools/run_sequence_loss_train.sh" "$method" 64 >"$run/train64.launcher.log" 2>&1
  test -f "$run/completed"
  test ! -f "$run/failed"
  test -f "$root/verl_checkpoints/seq_${method}_n4_seed11/global_step_64/actor/huggingface/config.json"
  cat "$run/train_2.log" "$run/train_64.log" >"$run/train_all_64.log"
  "$base/activevln_train_env/bin/python" "$root/tools/audit_training_gradients.py" "$run/train_all_64.log" 64 "$run/train_gradient_audit_64.json"
  printf "%s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ') completed 64 total optimizer steps for $method" >>"$state/sequence.log"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/training.completed"
