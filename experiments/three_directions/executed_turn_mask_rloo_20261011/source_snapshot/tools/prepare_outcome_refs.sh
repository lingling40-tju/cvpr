#!/usr/bin/env bash
set -Eeuo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_executed_turn_rloo_20261011"
sft="$base/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision_development/positive_initial_sft"
old="$base/ActiveVLN_three_direction_20261009/runlogs/development256"
result="$root/runlogs/development256"
mkdir -p "$result"
for spec in "positive_initial_sft:$sft" "td_turn_rloo_n4_seed11:$old/td_turn_rloo_n4_seed11"; do
  label=${spec%%:*}; source=${spec#*:}
  test -d "$source"
  if test -L "$result/$label"; then test "$(readlink "$result/$label")" = "$source"; else test ! -e "$result/$label"; ln -s "$source" "$result/$label"; fi
  if test "$label" = positive_initial_sft; then meta=positive_initial_sft; else meta=td_turn_rloo_n4_seed11; fi
  for suffix in completed validated.json dtype.json; do
    src="$old/$meta.$suffix"
    test -f "$src"
    if test -e "$result/$label.$suffix"; then
      cmp -s "$src" "$result/$label.$suffix" || { echo "reference artifact mismatch: $label.$suffix" >&2; exit 1; }
    else
      cp "$src" "$result/$label.$suffix"
    fi
  done
done
cd "$root"
"$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest prepared_data/development256.json --role development --label positive_initial_sft --output "$result/positive_initial_sft.preflight.json"
"$base/activevln_server_env/bin/python" tools/validate_train_label.py --root "$result" --manifest prepared_data/development256.json --role development --label td_turn_rloo_n4_seed11 --output "$result/td_turn_rloo_n4_seed11.preflight.json"
