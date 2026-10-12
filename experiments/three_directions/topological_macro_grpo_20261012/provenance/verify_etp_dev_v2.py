#!/usr/bin/env python3
import json, hashlib, math, os, random, statistics, datetime
ROOT='.'
MAN='prepared_data/development256_identity.json'
SFTROOT='/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision'
ETP='third_party/ETP-R1/data/logs/results/etp_r1_dev_eval_v2'
ARMS={
 'etp_dagger_init':('etp_r1_dagger_dev_v2',25000),
 'etp_macro_grpo64':('etp_r1_grpo64_dev_v2',64),
}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def load_raw(name,idx):
 d={}
 paths=[]
 for rank in range(4):
  p=os.path.join(ETP,name,'eval_results','stats_ep_ckpt_%d_development256_r%d_w4.json'%(idx,rank))
  assert os.path.isfile(p),p
  part=json.load(open(p)); paths.append(p)
  assert len(part)==64,(p,len(part))
  for k,v in part.items():
   assert k not in d,('duplicate',name,k)
   d[k]=v
 assert len(d)==256,(name,len(d))
 agg=json.load(open(os.path.join(ETP,name,'eval_results','stats_ckpt_%d_development256.json'%idx)))
 for k in agg:
  vals=[float(v[k]) for v in d.values()]
  assert all(math.isfinite(x) for x in vals),(name,k)
  actual=sum(vals)/len(vals)
  assert abs(actual-float(agg[k]))<1e-6,(name,k,actual,agg[k])
 return d,paths,agg
manifest=json.load(open(MAN)); ids=set(map(str,manifest['episode_ids'])); scenes=set(manifest['scene_ids'])
assert len(ids)==256 and manifest['scene_count']==8
verified=json.load(open(SFTROOT+'/verified_precision_comparisons.json'))
assert verified['actual_matched_sft_engine_dtype']=='float16'
assert verified['sft_decode_seed']==11 and verified['sft_training_seeds']==0
files=verified['screens']['development']['files_sha256']
compact=SFTROOT+'/development_with_sft.jsonl'; summary_path=SFTROOT+'/development_with_sft.json'; rec_path=SFTROOT+'/development_independent_recount.json'
for rel,p in [('runlogs/positive_matched_precision/development_with_sft.jsonl',compact),('runlogs/positive_matched_precision/development_with_sft.json',summary_path),('runlogs/positive_matched_precision/development_independent_recount.json',rec_path)]:
 assert files[rel]==sha(p),(rel,files[rel],sha(p))
summary=json.load(open(summary_path)); independent=json.load(open(rec_path))
evaltext=open('runlogs/freeze/eval_protocol_v2.json').read(); evalfreeze=json.loads(evaltext[:evaltext.rfind('}')+1])
assert summary['manifest_sha256']==evalfreeze['development_manifest_sha256']
assert summary['inference_errors']==0 and summary['episodes_per_model']==256
sft={}
for line in open(compact):
 x=json.loads(line); eid=str(x['episode_id']); assert eid not in sft
 sft[eid]={'scene_id':x['scene_id'],'success':float(x['models']['positive_initial_sft']['success']),'spl':float(x['models']['positive_initial_sft']['spl'])}
assert set(sft)==ids
assert set(x['scene_id'] for x in sft.values())==scenes
models={'positive_initial_sft':sft}; raw_files={}
for arm,(name,idx) in ARMS.items():
 d,paths,agg=load_raw(name,idx); raw_files[arm]=[{'path':p,'sha256':sha(p)} for p in paths]
 assert set(d)==ids
 models[arm]={eid:{'scene_id':sft[eid]['scene_id'],'success':float(v['success']),'spl':float(v['spl'])} for eid,v in d.items()}
 for eid,v in models[arm].items():
  assert v['success'] in (0.0,1.0) and math.isfinite(v['spl']) and 0.0<=v['spl']<=1.0,(arm,eid,v)
# scene-cluster percentile bootstrap over paired per-episode deltas
random.seed(20261012)
def comp(a,b):
 diffs={eid:{'sr':(models[a][eid]['success']-models[b][eid]['success'])*100.0,'spl':(models[a][eid]['spl']-models[b][eid]['spl'])*100.0} for eid in ids}
 byscene={}
 for eid,d in diffs.items(): byscene.setdefault(models[a][eid]['scene_id'],[]).append(d)
 sr=[sum(x['sr'] for x in vals)/len(vals) for vals in byscene.values()]
 spl=[sum(x['spl'] for x in vals)/len(vals) for vals in byscene.values()]
 bs_sr=[]; bs_spl=[]
 for _ in range(10000):
  ix=[random.randrange(len(sr)) for __ in range(len(sr))]
  bs_sr.append(sum(sr[i] for i in ix)/len(ix)); bs_spl.append(sum(spl[i] for i in ix)/len(ix))
 bs_sr.sort(); bs_spl.sort()
 mean=lambda key: sum(d[key] for d in diffs.values())/len(diffs)
 return {'paired_sr_pp':mean('sr'),'paired_spl_pp':mean('spl'),'scene_bootstrap_95pct_sr_pp':[bs_sr[249],bs_sr[9749]],'scene_bootstrap_95pct_spl_pp':[bs_spl[249],bs_spl[9749]],'episodes':len(diffs),'scenes':len(byscene),'scene_ids':sorted(byscene)}
comparisons={'etp_macro_grpo64_vs_etp_dagger_init':comp('etp_macro_grpo64','etp_dagger_init'),'etp_macro_grpo64_vs_fp16_qwen_sft':comp('etp_macro_grpo64','positive_initial_sft'),'etp_dagger_init_vs_fp16_qwen_sft':comp('etp_dagger_init','positive_initial_sft')}
absolute={}
for name,m in models.items(): absolute[name]={'episodes':len(m),'successes':int(sum(x['success'] for x in m.values())),'sr_percent':100*sum(x['success'] for x in m.values())/len(m),'spl_percent':100*sum(x['spl'] for x in m.values())/len(m)}
gate={'threshold_pp_each_metric':2.0,'candidate_vs_init_both_pass':comparisons['etp_macro_grpo64_vs_etp_dagger_init']['paired_sr_pp']>=2 and comparisons['etp_macro_grpo64_vs_etp_dagger_init']['paired_spl_pp']>=2,'candidate_vs_sft_both_pass':comparisons['etp_macro_grpo64_vs_fp16_qwen_sft']['paired_sr_pp']>=2 and comparisons['etp_macro_grpo64_vs_fp16_qwen_sft']['paired_spl_pp']>=2}
gate['expand_to_multiseed']=gate['candidate_vs_init_both_pass'] and gate['candidate_vs_sft_both_pass']
report={'schema':'etp_r1_dev256_independent_recount_v1','audited_utc':datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z','status':'PASS','manifest_sha256':evalfreeze['development_manifest_sha256'],'manifest_identity_file_sha256':sha(MAN),'manifest_episode_count':len(ids),'manifest_scene_count':len(scenes),'manifest_scenes':sorted(scenes),'sft_reference':{'freeze_sha256':verified['precision_freeze_sha256'],'matched_engine_dtype':verified['actual_matched_sft_engine_dtype'],'decode_seed':verified['sft_decode_seed'],'independent_training_seeds':verified['sft_training_seeds'],'compact_sha256':sha(compact),'summary_sha256':sha(summary_path),'upstream_independent_recount_sha256':sha(rec_path)},'raw_eval_files':raw_files,'absolute_metrics':absolute,'paired_comparisons':comparisons,'predeclared_gate':gate,'limitations':['development256 was reused/adaptive and is not a clean test set','single configured seed only','ETP macro-GRPO is compared both to its own DAgger initialization and to a separately trained FP16 Qwen SFT system','gate requires +2pp in paired SR and SPL versus both references; no multi-seed expansion unless all four comparisons pass']}
out='runlogs/grpo64_v2/evaluation_v2_recount.json'
with open(out,'x') as f: json.dump(report,f,indent=2,sort_keys=True)
print(json.dumps({'status':report['status'],'absolute_metrics':absolute,'paired_comparisons':comparisons,'gate':gate,'report_sha256':sha(out)},indent=2))
