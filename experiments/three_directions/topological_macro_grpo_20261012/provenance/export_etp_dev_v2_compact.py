#!/usr/bin/env python3
import json,hashlib,os,datetime
MAN='prepared_data/development256_identity.json'
SFTROOT='/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006/runlogs/positive_matched_precision'
OLDROOT='/Knowin/foundation/haozhiwang/whz/ActiveVLN_positive_trajectory_20261006'
ETP='third_party/ETP-R1/data/logs/results/etp_r1_dev_eval_v2'
REPORT='runlogs/grpo64_v2/evaluation_v2_recount.json'
TRAIN='runlogs/grpo64_v2/train_audit.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def jload(p):
 s=open(p).read(); return json.loads(s[:s.rfind('}')+1])
manifest=json.load(open(MAN)); expected=set(map(str,manifest['episode_ids']))
report=json.load(open(REPORT)); assert report['status']=='PASS' and report['manifest_episode_count']==256
# Load and independently verify the upstream FP16 SFT compact export identity.
ver=json.load(open(SFTROOT+'/verified_precision_comparisons.json'))
sc=ver['screens']['development']['files_sha256']
for rel,p in [('runlogs/positive_matched_precision/development_with_sft.jsonl',SFTROOT+'/development_with_sft.jsonl'),('runlogs/positive_matched_precision/development_with_sft.json',SFTROOT+'/development_with_sft.json'),('runlogs/positive_matched_precision/development_independent_recount.json',SFTROOT+'/development_independent_recount.json')]: assert sc[rel]==sha(p)
sft={}
for line in open(SFTROOT+'/development_with_sft.jsonl'):
 x=json.loads(line); eid=str(x['episode_id']); assert eid not in sft
 m=x['models']['positive_initial_sft']; sft[eid]={'scene_id':x['scene_id'],'success':float(m['success']),'spl':float(m['spl'])}
assert set(sft)==expected
models={'positive_initial_sft':sft}; raw=[]
for key,name,idx in [('etp_dagger_init','etp_r1_dagger_dev_v2',25000),('etp_macro_grpo64','etp_r1_grpo64_dev_v2',64)]:
 values={}
 for rank in range(4):
  p=os.path.join(ETP,name,'eval_results','stats_ep_ckpt_%d_development256_r%d_w4.json'%(idx,rank))
  dd=json.load(open(p)); assert len(dd)==64
  raw.append({'path':p,'sha256':sha(p)})
  for eid,v in dd.items(): assert eid not in values and eid in expected; values[eid]=v
 assert set(values)==expected
 models[key]={eid:{'scene_id':sft[eid]['scene_id'],'success':float(v['success']),'spl':float(v['spl'])} for eid,v in values.items()}
compact='runlogs/grpo64_v2/paired_episodes.jsonl'
with open(compact,'x') as f:
 for eid in sorted(expected,key=int):
  row={'episode_id':eid,'scene_id':sft[eid]['scene_id'],'models':{k:{'success':v[eid]['success'],'spl':v[eid]['spl']} for k,v in models.items()},'paired_delta_pp':{}}
  for a,b,label in [('etp_macro_grpo64','etp_dagger_init','etp_macro_grpo64_vs_etp_dagger_init'),('etp_macro_grpo64','positive_initial_sft','etp_macro_grpo64_vs_fp16_qwen_sft')]:
   row['paired_delta_pp'][label]={'sr':100*(models[a][eid]['success']-models[b][eid]['success']),'spl':100*(models[a][eid]['spl']-models[b][eid]['spl'])}
  f.write(json.dumps(row,sort_keys=True)+'\n')
# Verify compact membership and recomputed means against the independent raw report.
rows=[json.loads(line) for line in open(compact)]; assert len(rows)==256 and len(set(x['episode_id'] for x in rows))==256
for label in report['paired_comparisons']:
 vals=[x['paired_delta_pp'][label] for x in rows if label in x['paired_delta_pp']]
 if vals:
  assert abs(sum(v['sr'] for v in vals)/len(vals)-report['paired_comparisons'][label]['paired_sr_pp'])<1e-9
  assert abs(sum(v['spl'] for v in vals)/len(vals)-report['paired_comparisons'][label]['paired_spl_pp'])<1e-9
identity={'schema':'etp_r1_dev256_compact_identity_v1','created_utc':datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z','status':'PASS','development_manifest_sha256':report['manifest_sha256'],'development_manifest_identity_file_sha256':sha(MAN),'expected_episodes':256,'scenes':8,'train_audit_sha256':sha(TRAIN),'raw_recount_sha256':sha(REPORT),'paired_episode_export_sha256':sha(compact),'raw_eval_shards':raw,'sft_reference':{'precision_freeze_sha256':ver['precision_freeze_sha256'],'matched_engine_dtype':ver['actual_matched_sft_engine_dtype'],'decode_seed':ver['sft_decode_seed'],'independent_training_seeds':ver['sft_training_seeds'],'compact_sha256':sha(SFTROOT+'/development_with_sft.jsonl'),'summary_sha256':sha(SFTROOT+'/development_with_sft.json'),'upstream_independent_recount_sha256':sha(SFTROOT+'/development_independent_recount.json')},'metrics_recomputed_from_compact':'PASS','limitations':['development screen has been adaptively reused','single configured seed only']}
out='runlogs/grpo64_v2/compact_identity.json'
with open(out,'x') as f:json.dump(identity,f,indent=2,sort_keys=True)
print(json.dumps({'paired_episode_count':len(rows),'compact_sha256':identity['paired_episode_export_sha256'],'identity_sha256':sha(out)},indent=2))
