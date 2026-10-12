import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--sft-root',type=Path,required=True); p.add_argument('--candidate',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
manifest=a.root/'prepared_data/development256.json'; m=json.loads(manifest.read_text()); ids=[str(x) for x in m['episode_ids']]; scenes=[str(x) for x in m['scene_ids']]
assert len(ids)==len(set(ids))==256 and len(scenes)==256 and len(set(scenes))==8 and m['role']=='development'
def rows(root,label):
 out={}
 for shard in range(4):
  folder=root/label/f'shard_{shard:02d}'/'log'; expected={f'stats_{ids[i]}_0.json' for i in range(shard,256,4)}; observed={x.name for x in folder.glob('stats_*_0.json')}; assert observed==expected,(label,shard,len(expected),len(observed),list(expected^observed)[:3])
  for i in range(shard,256,4):
   eid=ids[i]; d=json.loads((folder/f'stats_{eid}_0.json').read_text()); assert str(d.get('episode_id',eid))==eid and d.get('early_stop_reason')!='inference_error'; s=d.get('success'); v=d.get('spl'); assert s in (0,0.0,1,1.0) and not isinstance(s,bool) and isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1; out[eid]=(int(s),float(v))
 assert set(out)==set(ids); return out
sft=rows(a.sft_root,'positive_initial_sft'); cand=rows(a.root/'runlogs/development256',a.candidate)
summary={}
for name,r in [('sft',sft),('candidate',cand)]: summary[name]={'episodes':len(r),'successes':sum(v[0] for v in r.values()),'sr':sum(v[0] for v in r.values())/256,'spl':sum(v[1] for v in r.values())/256}
summary['paired']={'sr_points':100*(summary['candidate']['sr']-summary['sft']['sr']),'spl_points':100*(summary['candidate']['spl']-summary['sft']['spl']),'candidate_only_success':sum(cand[e][0]>sft[e][0] for e in ids),'sft_only_success':sum(sft[e][0]>cand[e][0] for e in ids)}
summary.update({'schema':'prefix_bc_v4_independent_development_recount_v1','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'scenes':len(set(scenes)),'unique_episode_ids':len(set(ids)),'inference_errors':0,'scope':'Exploratory reused train-scene development screen; independent raw JSON recount, not clean test.'})
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f: json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary))
