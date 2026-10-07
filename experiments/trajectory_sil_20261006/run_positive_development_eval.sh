#!/usr/bin/env bash
set -euo pipefail
root=/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006
# The pair helper preserves the suite's existing control/candidate interface.
exec bash "$root/tools/run_positive_development_pair.sh" "$@"
