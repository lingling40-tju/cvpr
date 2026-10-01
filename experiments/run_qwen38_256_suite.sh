#!/usr/bin/env bash
set -euo pipefail

# Run three matched 256-step seed pairs using the original 8B parser and
# qwen3.8-max-0902 only for visual event verification.
root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930
run_dir="$root/runlogs/eventtrace_qwen38_r2r256"
prefix=eventtrace_qwen38_r2r256
proxy=http://127.0.0.1:5004
mkdir -p "$run_dir"
cd "$root"

healthy() {
  curl -fsS --max-time 5 "$proxy/health" | python3 -c \
    'import json,sys; x=json.load(sys.stdin); assert x["model"] == "qwen3.8-max-0902" and x["parser_ok"] and x["strict_transition"] is False and x["reasoning_effort"] is None'
  curl -fsS --max-time 5 http://127.0.0.1:5002/health >/dev/null
}

python3 - "$root/runlogs/eventtrace_audit_448_old/summary.json" \
  "$root/runlogs/eventtrace_audit_448_qwen38/summary.json" <<'PY'
import json, sys
old, new = (json.load(open(path)) for path in sys.argv[1:])
assert old["count"] == new["count"] == 49
assert old["annotation_csv_sha256"] == new["annotation_csv_sha256"]
assert new["completed_on_clear_ai_negative"] < old["completed_on_clear_ai_negative"]
print("Matched-resolution audit gate: clear-negative completions",
      old["completed_on_clear_ai_negative"], "->", new["completed_on_clear_ai_negative"])
PY
healthy

smoke="$root/verl_checkpoints/eventtrace_qwen38_smoke_seed11_event"
if [ ! -f "$run_dir/smoke.completed" ]; then
  if EVENTTRACE_EXPERIMENT_PREFIX=eventtrace_qwen38_smoke \
    EVENTTRACE_TOTAL_STEPS=2 EVENTTRACE_SAVE_FREQ=2 \
    EVENTTRACE_SEMANTIC_VERIFIER_URL="$proxy" \
    bash "$root/tools/run_multiseed_train.sh" 11 event >"$run_dir/smoke.log" 2>&1 && \
    python3 tools/check_grpo_preflight.py "$smoke/rollout.jsonl" \
      "$run_dir/smoke.log" --steps 2 >"$run_dir/smoke_grpo.json" && \
    python3 tools/check_semantic_run.py "$smoke/rollout.jsonl" \
      --steps 2 >"$run_dir/smoke_semantic.json"; then
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/smoke.completed"
  else
    status=$?
    printf '%s\n' "$status" >"$run_dir/smoke.failed"
    exit "$status"
  fi
fi

validate_arm() {
  local label=$1
  local arm=$2
  local rollout="$root/verl_checkpoints/${prefix}_${label}/rollout.jsonl"
  python3 tools/check_grpo_preflight.py "$rollout" "$run_dir/$label.log" \
    --steps 256 >"$run_dir/$label.grpo.json" 2>>"$run_dir/$label.log" || return
  if [ "$arm" = event ]; then
    python3 tools/check_semantic_run.py "$rollout" \
      --steps 256 >"$run_dir/$label.semantic.json" 2>>"$run_dir/$label.log" || return
  fi
}

for seed in 11 22 33; do
  for arm in control event; do
    label="seed${seed}_${arm}"
    if [ -f "$run_dir/$label.completed" ]; then continue; fi
    healthy
    date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.started"
    rm -f "$run_dir/$label.failed"
    curl -fsS --max-time 5 "$proxy/stats" >"$run_dir/$label.proxy_before.json"
    if EVENTTRACE_EXPERIMENT_PREFIX="$prefix" \
      EVENTTRACE_TOTAL_STEPS=256 EVENTTRACE_SAVE_FREQ=64 \
      EVENTTRACE_SEMANTIC_VERIFIER_URL="$proxy" \
      bash "$root/tools/run_multiseed_train.sh" "$seed" "$arm" \
        >"$run_dir/$label.log" 2>&1 && \
      validate_arm "$label" "$arm"; then
      curl -fsS --max-time 5 "$proxy/stats" >"$run_dir/$label.proxy_after.json"
      date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run_dir/$label.completed"
    else
      status=$?
      printf '%s\n' "$status" >"$run_dir/$label.failed"
      exit "$status"
    fi
  done
done

python3 tools/analyze_qwen38_training.py >"$run_dir/analysis.log"
