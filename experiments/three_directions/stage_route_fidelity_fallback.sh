#!/usr/bin/env bash
set -euo pipefail

# Prepare a separate source tree only after the progress experiment has a
# completed negative decision. This script does not launch GPU work.
base=/Knowin/foundation/haozhiwang/whz
source_root="$base/ActiveVLN_three_directions_20261002"
target="$base/ActiveVLN_route_fidelity_20261002"
progress_suite="$source_root/runlogs/three_direction_progress_scale_conditional"
progress_analysis="$source_root/runlogs/three_direction_full_val_unseen/scale_progress_128_analysis.json"
patch_file="$source_root/tools/terminal_ndtw_all_reasons.patch"

test -f "$progress_suite/suite.completed" || { echo 'progress experiment still active' >&2; exit 2; }
test ! -f "$progress_suite/suite.failed"
if ! test -f "$progress_suite/no_pilot_gain"; then
  test -s "$progress_analysis"
  "$base/activevln_server_env/bin/python" - "$progress_analysis" <<'PY'
import json, math, sys
d=json.load(open(sys.argv[1]))
assert d['split']=='val_unseen' and d['episodes']==1839
assert d['mode']=='progress' and d['train_steps']==128
assert set(d['paired_seed_differences'])=={'11','22','33'}
assert all(x['count']==1839 and x['inference_errors']==0 for x in d['models'].values())
sr,spl=d['mean_paired_sr_pp'],d['mean_paired_spl_pp']
assert math.isfinite(sr) and math.isfinite(spl)
assert not (sr>0 and spl>=0), 'progress gain retained; route fallback not eligible'
PY
fi
test ! -e "$target" || { echo "target already exists: $target" >&2; exit 2; }
test -s "$patch_file"
patch_sha=$(sha256sum "$patch_file" | awk '{print $1}')
[ "$patch_sha" = 5308e79541c767d04225d0792b65c8611d35529bb71b4fb2c8f9c0732ef5a705 ] || {
  echo 'route patch hash mismatch' >&2; exit 1;
}
source_sha=$(sha256sum "$source_root/vlnce_server/env.py" | awk '{print $1}')
[ "$source_sha" = 39a66e2b6e5e14839ba17360c6c1db9116ffd1623f02dd1dda8bb4ce920d107a ] || {
  echo 'base environment source changed' >&2; exit 1;
}

mkdir -p "$target"
for dir in vlnce_server verl examples tools; do cp -a "$source_root/$dir" "$target/$dir"; done
ln -s "$source_root/data" "$target/data"
mkdir -p "$target/runlogs" "$target/verl_checkpoints" "$target/outputs"
patch --dry-run -d "$target" -p1 <"$patch_file"
patch -d "$target" -p1 <"$patch_file"
"$base/activevln_train_env/bin/python" -m py_compile "$target/vlnce_server/env.py"
printf 'source=%s\nsource_env_sha256=%s\npatch_sha256=%s\n' \
  "$source_root" "$source_sha" "$patch_sha" >"$target/route_source.txt"
echo "$target"
