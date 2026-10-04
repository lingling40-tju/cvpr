#!/usr/bin/env bash
set -euo pipefail

# Public Route2Step MIA checkpoint. Only the MIA module is needed for the
# frozen offline semantic-progress screen. We do not publish the weights.
base=/Knowin/foundation/haozhiwang/whz
run="$base/route2step_mia_20261004"
revision=a2abfa61a0a75949779e4fb1aea12bab7cbf770f
mkdir -p "$run/runlogs/download" "$run/hf_home" "$run/model"
exec 9>"$run/runlogs/download/run.lock"
flock -n 9 || { echo 'Route2Step MIA download already active' >&2; exit 2; }
if test -f "$run/runlogs/download/completed"; then exit 0; fi
rm -f "$run/runlogs/download/failed"
on_exit() { status=$?; if test "$status" -ne 0; then echo "$status" >"$run/runlogs/download/failed"; fi; }
trap on_exit EXIT
export HF_HOME="$run/hf_home"
export HF_HUB_CACHE="$run/hf_home/hub"
export HF_XET_CACHE="$run/hf_home/xet"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=30
"$base/activevln_train_env/bin/python" - "$run/model" "$revision" <<'PY'
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id="XiangyunHuang/Route2Step",
    revision=sys.argv[2],
    allow_patterns=["MIA/*"],
    local_dir=sys.argv[1],
    max_workers=2,
)
PY
for name in config.json model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  preprocessor_config.json tokenizer.json; do
  test -s "$run/model/MIA/$name"
done
sha256sum "$run/model/MIA/config.json" \
  "$run/model/MIA/model.safetensors.index.json" \
  "$run/model/MIA/model-00001-of-00002.safetensors" \
  "$run/model/MIA/model-00002-of-00002.safetensors" \
  "$run/model/MIA/preprocessor_config.json" \
  "$run/model/MIA/tokenizer.json" \
  >"$run/runlogs/download/model.sha256"
"$base/activevln_train_env/bin/python" - "$run/runlogs/download" "$revision" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
report = {
    "schema": "route2step_mia_download_v1",
    "repository": "XiangyunHuang/Route2Step",
    "revision": sys.argv[2],
    "component": "MIA",
    "upstream_code": "https://github.com/BUAA-GAMMA-LAB/Route2Step",
    "interpretation": "Checkpoint provenance only; no offline score or navigation result",
}
(root / "model_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$run/runlogs/download/completed"
