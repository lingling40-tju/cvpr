#!/usr/bin/env python3
import argparse, hashlib, json, math, statistics
from pathlib import Path

MANIFEST_SHA="8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3"
LABELS=("positive_initial_sft","online_dagger_control_64step_seed11","online_dagger_candidate_64step_seed11")

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load_label(root,label,ids):
    found={}
    hashes={}
    for shard in range(4):
        folder=root/label/f"shard_{shard:02d}"/"log"
        expected=[ids[i] for i in range(shard,len(ids),4)]
        actual={p.name for p in folder.glob("stats_*_0.json")}
        if actual!={f"stats_{i}_0.json" for i in expected}:
            raise ValueError(f"coverage mismatch {label} shard {shard}")
        for eid in expected:
            p=folder/f"stats_{eid}_0.json"; row=json.loads(p.read_text())
            if str(row.get("episode_id",eid))!=eid: raise ValueError(f"id mismatch {label} {eid}")
            if row.get("early_stop_reason")=="inference_error": raise ValueError(f"inference error {label} {eid}")
            ok=row.get("success"); spl=row.get("spl")
            if isinstance(ok,bool): ok=int(ok)
            if ok not in (0,0.0,1,1.0) or isinstance(spl,bool) or not isinstance(spl,(int,float)) or not math.isfinite(spl) or not 0<=spl<=1:
                raise ValueError(f"invalid metric {label} {eid}")
            found[eid]={"success":int(ok),"spl":float(spl)}
            hashes[eid]=sha(p)
    if set(found)!=set(ids): raise ValueError(f"unique ID coverage mismatch {label}")
    return found,hashes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--sft-root",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--compact",type=Path,required=True)
    a=ap.parse_args()
    assert sha(a.manifest)==MANIFEST_SHA
    m=json.loads(a.manifest.read_text()); ids=[str(x) for x in m["episode_ids"]]; scenes=[str(x) for x in m["scene_ids"]]
    assert m["schema"]=="trajectory_sil_rl_train_screen_v1" and m["split"]=="train" and m["role"]=="development"
    assert len(ids)==len(set(ids))==256 and len(scenes)==256 and len(set(scenes))==8
    datasets={}
    for label,root in (("positive_initial_sft",a.sft_root),("online_dagger_control_64step_seed11",a.root),("online_dagger_candidate_64step_seed11",a.root)):
        datasets[label],hashes=load_label(root,label,ids)
        datasets[label+"_raw_sha256_by_episode"]=hashes
    rows=[]
    for eid,scene in zip(ids,scenes):
        row={"episode_id":eid,"scene_id":scene}
        for label in LABELS:
            val=datasets[label][eid]
            row[label+"_success"]=val["success"]; row[label+"_spl"]=val["spl"]
        rows.append(row)
    def summary(label):
        return {"successes":sum(datasets[label][i]["success"] for i in ids),
                "sr":sum(datasets[label][i]["success"] for i in ids)/256,
                "spl":statistics.mean(datasets[label][i]["spl"] for i in ids),
                "episodes":256,"unique_ids":256,"inference_errors":0}
    s={label:summary(label) for label in LABELS}
    def delta(arm):
        return {"paired_sr_points":100*(s[arm]["sr"]-s["positive_initial_sft"]["sr"]),
                "paired_spl_points":100*(s[arm]["spl"]-s["positive_initial_sft"]["spl"]),
                "sr_gate_pass":100*(s[arm]["sr"]-s["positive_initial_sft"]["sr"])>=2.0,
                "spl_gate_pass":100*(s[arm]["spl"]-s["positive_initial_sft"]["spl"])>=2.0}
    result={"schema":"online_dagger_dev_raw_independent_recount_v1","role":"development","split":"train",
            "manifest_sha256":MANIFEST_SHA,"episodes":256,"scenes":8,"models":s,
            "candidate_vs_sft":delta("online_dagger_candidate_64step_seed11"),
            "control_vs_sft":delta("online_dagger_control_64step_seed11"),
            "candidate_vs_control":{"paired_sr_points":100*(s["online_dagger_candidate_64step_seed11"]["sr"]-s["online_dagger_control_64step_seed11"]["sr"]),
                "paired_spl_points":100*(s["online_dagger_candidate_64step_seed11"]["spl"]-s["online_dagger_control_64step_seed11"]["spl"])},
            "raw_stat_sha256_by_label":{k:v for k,v in datasets.items() if k.endswith("_raw_sha256_by_episode")},
            "claim_scope":"Independent CPU recount of raw Habitat per-episode stats; one configured seed on an adaptively reused development split."
    }
    a.compact.parent.mkdir(parents=True,exist_ok=True)
    a.compact.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    result["compact_sha256"]=sha(a.compact)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"pass":True,"episodes":256,"scenes":8,"candidate_vs_sft":result["candidate_vs_sft"],"candidate_vs_control":result["candidate_vs_control"]}))
if __name__=="__main__": main()
