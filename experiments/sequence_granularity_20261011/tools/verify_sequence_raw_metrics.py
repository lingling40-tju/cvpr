"""Independent internal-ID and navigation-metric audit over raw development stats."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--methods",nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    assert sha(a.manifest)=="8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3"
    m=json.loads(a.manifest.read_text())
    ids=[str(x) for x in m["episode_ids"]]
    scenes=[str(x) for x in m["scene_ids"]]
    assert len(ids)==len(set(ids))==256 and len(set(scenes))==8
    labels=["positive_initial_sft"]+["seq_"+x+"_n4_seed11" for x in a.methods]
    records={}
    digests={}
    for label in labels:
        assert (a.root/(label+".completed")).is_file()
        assert not (a.root/(label+".failed")).exists()
        rows={}
        for shard in range(4):
            folder=a.root/label/f"shard_{shard:02d}"/"log"
            expected={f"stats_{eid}_0.json" for eid in ids[shard::4]}
            observed={x.name for x in folder.glob("stats_*_0.json")}
            assert observed==expected,(label,shard,len(observed),len(expected))
            for eid in ids[shard::4]:
                f=folder/f"stats_{eid}_0.json"
                d=json.loads(f.read_text())
                assert str(d.get("id"))==eid,(label,eid,d.get("id"))
                assert str(d.get("episode_id",eid))==eid,(label,eid,d.get("episode_id"))
                assert d.get("early_stop_reason")!="inference_error",(label,eid)
                s=d["success"]; spl=d["spl"]
                assert s in (False,True,0,1,0.0,1.0)
                assert not isinstance(spl,bool) and isinstance(spl,(int,float)) and math.isfinite(spl) and 0<=spl<=1
                rows[eid]=(int(bool(s)),float(spl))
                digests[f"{label}/shard_{shard:02d}/{f.name}"]=sha(f)
        assert set(rows)==set(ids)
        valid=json.loads((a.root/(label+".validated.json")).read_text())
        assert valid["episodes"]==256 and valid["inference_errors"]==0
        assert valid["successes"]==sum(x[0] for x in rows.values())
        assert math.isclose(valid["spl"],sum(x[1] for x in rows.values())/256,abs_tol=1e-10,rel_tol=0)
        records[label]=rows
    pairs={}
    for method in a.methods:
        label="seq_"+method+"_n4_seed11"
        sr=100*sum(records[label][eid][0]-records["positive_initial_sft"][eid][0] for eid in ids)/256
        spl=100*sum(records[label][eid][1]-records["positive_initial_sft"][eid][1] for eid in ids)/256
        pairs[method]={"sft_sr_points":sr,"sft_spl_points":spl,
                       "candidate_successes":sum(records[label][eid][0] for eid in ids),
                       "sft_successes":sum(records["positive_initial_sft"][eid][0] for eid in ids)}
        report=json.loads((a.root.parent/"sequence_evaluation"/(method+"_vs_sft.json")).read_text())
        assert math.isclose(sr,report["paired_sr_points"],abs_tol=1e-10,rel_tol=0)
        assert math.isclose(spl,report["paired_spl_points"],abs_tol=1e-10,rel_tol=0)
    out={"schema":"sequence_granularity_raw_internal_id_recount_v1","status":"PASS",
         "episodes_per_model":256,"models":len(labels),"raw_ids_checked":256*len(labels),
         "scenes":8,"inference_errors":0,"comparisons":pairs,
         "raw_stats_sha256":digests,"manifest_sha256":sha(a.manifest),
         "verifier_sha256":sha(Path(__file__)),
         "scope":"Independent raw per-episode ID and SR/SPL recount; one reused development screen, one configured seed."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:f.write(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="raw_stats_sha256"}))


if __name__=="__main__":
    main()
