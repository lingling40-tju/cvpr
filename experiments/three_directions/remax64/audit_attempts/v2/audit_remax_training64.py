#!/usr/bin/env python3
import argparse, hashlib, json, math, re
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ap=argparse.ArgumentParser()
ap.add_argument("--root",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
a=ap.parse_args(); root=a.root; train=root/"runlogs/remax_n4_seed11"
events=sorted((train/"tensorboard").glob("events.out.tfevents.*"))
assert len(events)==3, [p.name for p in events]
required=("actor/grad_norm","actor/ppo_kl","timing_s/gen_max","critic/rewards/mean","critic/advantages/mean")
values={k:{} for k in required}; event_hashes={}
for p in events:
    event_hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    ea=EventAccumulator(str(p),size_guidance={"scalars":0}); ea.Reload()
    for tag in required:
        if tag not in ea.Tags().get("scalars",[]): continue
        for e in ea.Scalars(tag):
            if e.step in values[tag]: raise AssertionError(f"duplicate scalar {tag} step {e.step}")
            values[tag][int(e.step)]=float(e.value)
expected=set(range(1,65))
for tag in required:
    assert set(values[tag])==expected,(tag,sorted(values[tag]))
    assert all(math.isfinite(v) for v in values[tag].values()),tag
assert all(v>0 for v in values["actor/grad_norm"].values())
assert all(v>0 for v in values["timing_s/gen_max"].values())
smoke=json.loads((train/"smoke_gradient_audit.json").read_text())
assert smoke["expected_steps"]==smoke["observed_steps"]==smoke["nonzero_gradient_steps"]==2
assert smoke["zero_or_rounded_gradient_step_ids"]==[]
log=(train/"train.log").read_text(errors="replace")
ansi=re.compile(r"\x1b\[[0-9;]*m")
rows={}
for raw in log.splitlines():
    line=ansi.sub("",raw); m=re.search(r"step:(\d+) - global_seqlen",line)
    if not m: continue
    step=int(m.group(1)); assert step not in rows
    def get(name):
        z=re.search(re.escape(name)+r":([-+0-9.eE]+)",line)
        if not z: raise AssertionError(f"missing {name} at logged step {step}")
        return float(z.group(1))
    rows[step]={"actor/grad_norm":get("actor/grad_norm"),"actor/ppo_kl":get("actor/ppo_kl"),
        "timing_s/gen_max":get("timing_s/gen_max"),"critic/rewards/mean":get("critic/rewards/mean"),
        "critic/advantages/mean":get("critic/advantages/mean")}
assert set(rows)==set(range(3,65)),(len(rows),sorted(rows)[:4],sorted(rows)[-4:])
for step,row in rows.items():
    for tag,v in row.items():
        assert math.isfinite(v)
        assert abs(v-values[tag][step])<=0.0011+abs(v)*1e-6,(step,tag,v,values[tag][step])
gradients={step:v for step,v in values["actor/grad_norm"].items()}
kl={step:v for step,v in values["actor/ppo_kl"].items()}
assert all(v>0 for v in gradients.values()) and all(math.isfinite(v) for v in kl.values())
report={"schema":"remax_training_combined_smoke_resume_audit_v2","total_optimizer_updates":64,
 "smoke_updates":2,"resumed_updates":62,"step_ids":sorted(expected),
 "actor_gradient":{"steps":64,"nonzero_steps":sum(v>0 for v in gradients.values()),"min":min(gradients.values()),"max":max(gradients.values())},
 "finite_kl_steps":64,"greedy_baseline_steps":64,"greedy_baseline_seconds_min":min(values["timing_s/gen_max"].values()),
 "resumed_console_crosscheck_steps":62,"scalar_tags":{k:{"steps":64,"min":min(d.values()),"max":max(d.values())} for k,d in values.items()},
 "smoke_gradient_audit_sha256":hashlib.sha256((train/"smoke_gradient_audit.json").read_bytes()).hexdigest(),
 "smoke_train_log_sha256_recorded":smoke["train_log_sha256"],
 "resume_train_log_sha256":hashlib.sha256((train/"train.log").read_bytes()).hexdigest(),
 "tensorboard_event_sha256":event_hashes}
a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("REMAX_TRAIN_COMBINED_AUDIT_PASS 64 updates; 2 smoke + 62 resumed")
