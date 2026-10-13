#!/usr/bin/env bash
set -euo pipefail
base=/Knowin/foundation/haozhiwang/whz
root="$base/ActiveVLN_online_dagger_action_supervision_20261012"
state="$root/runlogs/online_dagger_pilot64"
mkdir -p "$state"
exec 9>"$state/suite.lock"; flock -n 9 || { echo "pilot suite already active" >&2; exit 2; }
if test -f "$state/suite.completed"; then exit 0; fi
test ! -f "$state/suite.failed" || { echo "prior pilot failure marker exists" >&2; exit 2; }
on_exit() {
 status=$?
 if test "$status" -ne 0; then printf '%s\n' "$status" >"$state/suite.failed"; fi
 if test -s "$root/runlogs/dagger_service/server.pid"; then
  pid=$(cat "$root/runlogs/dagger_service/server.pid")
  if ps -o args= -p "$pid" 2>/dev/null | grep -F 'server.port=5087' >/dev/null; then bash "$root/tools/dagger_service_stop.sh" || true; fi
 fi
}
trap on_exit EXIT
while true; do
 busy=0
 for gpu in 0 1; do mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits); if test "$mem" -ge 10000; then busy=1; fi; done
 mem2=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits); if test "$mem2" -ge 42000; then busy=1; fi
 if test "$busy" -eq 0; then break; fi
 sleep 30
done
python3 - "$base/models/Qwen2.5-VL-3B_sft_r2r_envdrop_multiturn" "$root/prepared_data/positive_initial_sft_identity.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
model=Path(sys.argv[1]); identity=json.loads(Path(sys.argv[2]).read_text())
assert identity["rl_optimizer_steps"]==0
for name,meta in identity["files"].items():
 p=model/name; assert p.stat().st_size==meta["bytes"],name
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 assert h.hexdigest()==meta["sha256"],name
print("SFT identity verified",len(identity["files"]),"files")
PY
cp "$root/prepared_data/positive_initial_sft_identity.json" "$state/sft_identity.json"
sha256sum "$root/verl/workers/agent/parallel_env_vlnce.py" "$root/verl/workers/actor/dp_actor.py" "$root/vlnce_server/env.py" "$root/vlnce_server/semantic_reward/env.py" "$root/tools/dagger_train_pilot.sh" "$root/tools/audit_dagger_pilot.py" >"$state/source_sha256.txt"
! curl -fsS --max-time 1 http://127.0.0.1:5087/health >/dev/null 2>&1
bash "$root/tools/dagger_service_start.sh" >"$state/service_start.log" 2>&1
for arm in control candidate; do
 label="online_dagger_${arm}_64step_seed11"
 if ! test -f "$state/${arm}_train_audit.json"; then
   bash "$root/tools/dagger_train_pilot.sh" "$arm" >"$state/${arm}_launcher.log" 2>&1
   python3 "$root/tools/audit_dagger_pilot.py" --root "$root" --arm "$arm" --output "$state/${arm}_train_audit.json" >"$state/${arm}_audit.log"
 fi
 test -f "$root/runlogs/$label/completed"
done
bash "$root/tools/dagger_service_stop.sh" >"$state/service_stop.log" 2>&1
python3 - "$state" "$root/prepared_data/positive_initial_sft_identity.json" "$root/prepared_data/fit512.parquet" <<'PY'
import hashlib,json,sys
from pathlib import Path
state=Path(sys.argv[1]);sft=json.loads(Path(sys.argv[2]).read_text());data=Path(sys.argv[3])
c=json.loads((state/"candidate_train_audit.json").read_text());r=json.loads((state/"control_train_audit.json").read_text())
assert c["optimizer_steps"]==r["optimizer_steps"]==64 and c["teacher_examples"]==r["teacher_examples"]==2048
assert c["arm"]=="candidate" and r["arm"]=="control"
freeze={"schema":"online_dagger_pilot_freeze_v1","fit512_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),"sft_identity_sha256":hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),"sft_optimizer_steps":sft["rl_optimizer_steps"],"candidate":c,"control":r,"evaluation_manifest_sha256":"8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3","gate":"candidate vs same-FP16 SFT: paired SR and SPL each >= +2.0pp; candidate vs RLOO control and historical GRPO are attribution/context only"}
p=state/"pilot_train_freeze.json";assert not p.exists();p.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n")
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$state/suite.completed"

