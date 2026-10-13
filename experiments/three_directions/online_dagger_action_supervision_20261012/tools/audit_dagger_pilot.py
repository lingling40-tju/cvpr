#!/usr/bin/env python3
import argparse, json, math, hashlib
from collections import Counter
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LABELS={0:"stop",1:"move forward 25cm",2:"turn left 15 degrees",3:"turn right 15 degrees"}

def scalars(path):
    acc=EventAccumulator(str(path),size_guidance={"scalars":0})
    acc.Reload()
    return {tag:[float(v.value) for v in acc.Scalars(tag)] for tag in acc.Tags().get("scalars",[])}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--arm",choices=("candidate","control"),required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    root=Path(args.root)
    exp=f"online_dagger_{args.arm}_64step_seed11"
    run=root/"runlogs"/exp
    ckpt=root/"verl_checkpoints"/exp
    assert (run/"completed").is_file() and not (run/"failed").exists()
    assert (ckpt/"global_step_64"/"actor"/"huggingface"/"config.json").is_file()
    rollout=ckpt/"rollout.jsonl"
    raw_sha=hashlib.sha256(rollout.read_bytes()).hexdigest()
    by_step={}
    for line in rollout.read_text().splitlines():
        if line.strip():
            row=json.loads(line)
            by_step.setdefault(int(row["step"]),[]).extend(row["info"])
    assert sorted(by_step)==list(range(1,65)), f"expected steps 1..64, got {sorted(by_step)}"
    updates=[]
    for step,infos in sorted(by_step.items()):
        assert len(infos)==32, f"update {step}: expected 32 trajectories, got {len(infos)}"
        audits=[x.get("dagger_example_audit") for x in infos]
        assert all(isinstance(x,dict) for x in audits), f"update {step}: missing teacher audit"
        assert all(x.get("target_mask_nonempty") and x.get("target_action_tokens",0)>0 and x.get("context_tokens",0)>0 for x in audits)
        assert all(x.get("teacher_action_id") in LABELS and x.get("teacher_text")==LABELS[x["teacher_action_id"]] for x in audits)
        groups=Counter(str(x["episode_id"]) for x in infos)
        assert len(groups)==8 and set(groups.values())=={4}, f"update {step}: invalid 8x4 groups"
        updates.append({"step":step,"trajectories":32,"groups":8,
            "labels":dict(Counter(x["teacher_text"] for x in audits)),
            "nonempty_action_masks":sum(bool(x["target_mask_nonempty"]) for x in audits),
            "teacher_action_tokens":sum(int(x["target_action_tokens"]) for x in audits)})
    event_files=list((run/"tensorboard").glob("**/events.out.tfevents.*"))
    assert event_files, "TensorBoard scalar files missing"
    metrics={}
    for f in event_files:
        for key,values in scalars(f).items():
            metrics.setdefault(key,[]).extend(values)
    def find(fragment):
        return [v for key,values in metrics.items() if fragment in key for v in values]
    grad,kl=find("actor/grad_norm"),find("actor/kl_loss")
    assert len(grad)==64 and all(math.isfinite(v) and v>0 for v in grad), f"bad actor-gradient coverage {len(grad)}"
    assert len(kl)==64 and all(math.isfinite(v) for v in kl), f"bad KL coverage {len(kl)}"
    result={"schema":"online_dagger_pilot_train_audit_v1","arm":args.arm,
        "optimizer_steps":64,"teacher_examples":2048,"valid_target_coverage":1.0,
        "rollout_sha256":raw_sha,"updates":updates,"actor_grad_norm_values":grad,"kl_values":kl}
    if args.arm=="candidate":
        ce,count,tokens=find("actor/dagger_ce_loss"),find("actor/dagger_num_examples"),find("actor/dagger_target_tokens")
        assert len(ce)==64 and all(math.isfinite(v) and v>0 for v in ce)
        assert len(count)==64 and all(math.isfinite(v) and v>0 for v in count)
        assert len(tokens)==64 and all(math.isfinite(v) and v>0 for v in tokens)
        result.update({"dagger_ce_values":ce,"dagger_num_examples_metrics":count,"dagger_target_token_metrics":tokens})
    else:
        assert not any("actor/dagger_ce_loss" in key for key in metrics), "control unexpectedly trained DAgger CE"
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists(), f"refusing to overwrite {out}"
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"audit":str(out),"pass":True,"steps":64,"teacher_examples":2048,"rollout_sha256":raw_sha}))

if __name__=="__main__":
    main()

