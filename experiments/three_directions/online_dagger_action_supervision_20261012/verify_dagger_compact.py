#!/usr/bin/env python3
"""Independently recount the compact per-episode DAgger development export."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
EXPECTED="8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3"
LABELS=("positive_initial_sft","online_dagger_control_64step_seed11","online_dagger_candidate_64step_seed11")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--root",type=Path,default=Path(__file__).parent/"runlogs/online_dagger_development256")
 ap.add_argument("--output",type=Path)
 a=ap.parse_args(); manifest=Path(__file__).parent/"prepared_data/development256.json"; compact=a.root/"independent_episodes.jsonl"
 assert sha(manifest)==EXPECTED
 m=json.loads(manifest.read_text()); ids=[str(x) for x in m["episode_ids"]]; scenes=[str(x) for x in m["scene_ids"]]
 assert len(ids)==len(set(ids))==256 and len(set(scenes))==8
 rows=[json.loads(x) for x in compact.read_text().splitlines() if x]
 assert len(rows)==256 and [str(r["episode_id"]) for r in rows]==ids and [str(r["scene_id"]) for r in rows]==scenes
 stats={}
 for label in LABELS:
  sk,pk=label+"_success",label+"_spl"
  assert all(r[sk] in (0,1) for r in rows)
  assert all(isinstance(r[pk],(int,float)) and math.isfinite(r[pk]) and 0<=r[pk]<=1 for r in rows)
  n=sum(r[sk] for r in rows); stats[label]={"successes":n,"sr":n/256,"spl":statistics.mean(r[pk] for r in rows),"episodes":256,"unique_ids":256,"inference_errors":0}
 def pair(a,b): return {"paired_sr_points":100*(stats[a]["sr"]-stats[b]["sr"]),"paired_spl_points":100*(stats[a]["spl"]-stats[b]["spl"])}
 out={"schema":"online_dagger_dev_compact_independent_recount_v1","manifest_sha256":EXPECTED,"episodes":256,"scenes":8,"models":stats,"candidate_vs_sft":pair(LABELS[2],LABELS[0]),"control_vs_sft":pair(LABELS[1],LABELS[0]),"candidate_vs_control":pair(LABELS[2],LABELS[1]),"candidate_gate_pass":pair(LABELS[2],LABELS[0])["paired_sr_points"]>=2 and pair(LABELS[2],LABELS[0])["paired_spl_points"]>=2,"source_compact_sha256":sha(compact),"scope":"One configured seed on adaptively reused development scenes; descriptive only."}
 remote=json.loads((a.root/"final_report.json").read_text())
 for key in ("candidate_vs_sft","control_vs_sft","candidate_vs_control"):
  for metric in ("paired_sr_points","paired_spl_points"): assert abs(out[key][metric]-remote[key][metric])<1e-9,(key,metric)
 assert out["models"]==remote["models"] and out["candidate_gate_pass"]==remote["gate_pass"]
 out["remote_report_sha256"]=sha(a.root/"final_report.json")
 target=a.output or a.root/"independent_compact_recount.json"; target.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"pass":True,"candidate_gate_pass":out["candidate_gate_pass"],"models":stats,"candidate_vs_sft":out["candidate_vs_sft"]},indent=2))
if __name__=="__main__": main()
