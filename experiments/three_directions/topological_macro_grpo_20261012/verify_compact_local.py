#!/usr/bin/env python3
"""Independent stdlib-only arithmetic check for the archived ETP dev256 compact rows."""
import hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'reports'
rows=[json.loads(x) for x in (R/'paired_episodes.jsonl').read_text().splitlines() if x]
assert len(rows)==256
ids=[str(x['episode_id']) for x in rows]
assert len(set(ids))==256
scenes={x['scene_id'] for x in rows}
assert len(scenes)==8
report=json.loads((R/'evaluation_v2_recount.json').read_text())
identity=json.loads((R/'compact_identity.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'paired_episodes.jsonl')==identity['paired_episode_export_sha256']
assert sha(R/'evaluation_v2_recount.json')==identity['raw_recount_sha256']
assert identity['expected_episodes']==len(rows) and identity['scenes']==len(scenes)
models=('positive_initial_sft','etp_dagger_init','etp_macro_grpo64')
absolute={}
for m in models:
 succ=[]; spl=[]
 for row in rows:
  x=row['models'][m]; s=float(x['success']); p=float(x['spl'])
  assert s in (0.0,1.0) and math.isfinite(p) and 0.0<=p<=1.0
  succ.append(s); spl.append(p)
 absolute[m]={'successes':int(sum(succ)),'episodes':len(rows),'sr_percent':100*sum(succ)/len(rows),'spl_percent':100*sum(spl)/len(rows)}
for m,x in absolute.items():
 y=report['absolute_metrics'][m]
 assert x['successes']==y['successes'] and abs(x['sr_percent']-y['sr_percent'])<1e-12 and abs(x['spl_percent']-y['spl_percent'])<1e-10
paired={}
for label in ('etp_macro_grpo64_vs_etp_dagger_init','etp_macro_grpo64_vs_fp16_qwen_sft','etp_dagger_init_vs_fp16_qwen_sft'):
 if all(label in row['paired_delta_pp'] for row in rows):
  vals=[row['paired_delta_pp'][label] for row in rows]
 else:
  assert label=='etp_dagger_init_vs_fp16_qwen_sft'
  vals=[{'sr':100*(row['models']['etp_dagger_init']['success']-row['models']['positive_initial_sft']['success']),'spl':100*(row['models']['etp_dagger_init']['spl']-row['models']['positive_initial_sft']['spl'])} for row in rows]
 assert len(vals)==256
 a=sum(float(x['sr']) for x in vals)/len(vals); b=sum(float(x['spl']) for x in vals)/len(vals)
 y=report['paired_comparisons'][label]
 assert abs(a-y['paired_sr_pp'])<1e-10 and abs(b-y['paired_spl_pp'])<1e-10
 paired[label]={'paired_sr_pp':a,'paired_spl_pp':b}
gate={'threshold_pp_each_metric':2.0,'candidate_vs_init_both_pass':paired['etp_macro_grpo64_vs_etp_dagger_init']['paired_sr_pp']>=2 and paired['etp_macro_grpo64_vs_etp_dagger_init']['paired_spl_pp']>=2,'candidate_vs_sft_both_pass':paired['etp_macro_grpo64_vs_fp16_qwen_sft']['paired_sr_pp']>=2 and paired['etp_macro_grpo64_vs_fp16_qwen_sft']['paired_spl_pp']>=2}
gate['expand_to_multiseed']=gate['candidate_vs_init_both_pass'] and gate['candidate_vs_sft_both_pass']
assert gate==report['predeclared_gate']
assert absolute['positive_initial_sft']['successes']==109
result={'schema':'etp_macro_dev256_local_compact_recount_v1','status':'PASS','independent_implementation':'Python standard library only; reads compact per-episode outcomes and recomputes all means and gate; imports no remote analyzer','episodes':len(rows),'scenes':len(scenes),'absolute_metrics':absolute,'paired_metrics':paired,'gate':gate,'limitations':['single configured seed','adaptively reused development screen; not a clean test','does not independently re-read remote raw shard files']}
saved=json.loads((R/'local_compact_recount.json').read_text())
assert saved==result
print(json.dumps(result,indent=2,sort_keys=True))
