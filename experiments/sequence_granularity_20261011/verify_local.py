#!/usr/bin/env python3
"""Independent compact recount for the three sequence-level loss pilots."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load(rel): return json.loads((ROOT/rel).read_text())
def sha(rel): return hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
manifest_rel='prepared_data/development256.json'
manifest=load(manifest_rel)
manifest_sha='8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3'
assert sha(manifest_rel)==manifest_sha
episode_ids=[str(x) for x in manifest['episode_ids']]
assert len(episode_ids)==len(set(episode_ids))==256
assert len(set(manifest['scene_ids']))==8
train_id='runlogs/sequence_training/freeze_identity.json'
assert sha(train_id)=='fb582d292763d16beae50a6a5ddfccd8874408f8a25d6e316bad86620508146d'
training=load(train_id)
assert training['configured_seed']==11 and training['group_size']==4
assert training['optimizer_updates_total_including_smoke']==64
assert training['train_rows']==512 and training['recorded_trajectory_budget_per_method']==2048
assert sha('verl/trainer/ppo/sequence_policy_losses.py')=='59bcc0b89bc74db5156be0545b35115b15d83b573f94690fa04e03f790cb1400'
assert sha('verl/trainer/ppo/core_algos.py')==training['core_algos_sha256']
assert sha('tools/run_sequence_loss_train.sh')==training['train_launcher_sha256']
eval_id_rel='runlogs/sequence_evaluation/evaluation_identity.json'
eval_id_sha='89cd425e4d697dbd9a90657624b213727667b617e43f11fd4e1fe807715bad5c'
assert sha(eval_id_rel)==eval_id_sha
assert sha('tools/verify_sequence_raw_metrics.py')=='d92d8abd2c65bb61844e10a859b7e91dbf6dfc8aa5a8fb25c0254b59185e97f5'
report=load('runlogs/sequence_evaluation/sequence_granularity_development_report.json')
assert report['manifest_sha256']==manifest_sha and report['episodes']==256 and report['scenes']==8
assert report['inference_dtype']=='float16' and report['same_fp16_sft_reference'] is True
raw=load('runlogs/sequence_evaluation/raw_id_independent_recount.json')
assert raw['status']=='PASS' and raw['raw_ids_checked']==1024
assert raw['episodes_per_model']==256 and raw['models']==4 and raw['scenes']==8 and raw['inference_errors']==0
assert len(raw['raw_stats_sha256'])==1024
assert raw['manifest_sha256']==manifest_sha
assert (ROOT/'runlogs/sequence_evaluation/suite.completed').is_file()
assert (ROOT/'runlogs/sequence_training/training.completed').is_file()

pairs={
 'gspo_sequence':('gspo_sequence_vs_sft.jsonl','gspo_sequence_vs_sft.json','gspo_sequence_vs_sft.independent.json',53,-21.875,-20.560268140284066),
 'gmpo_geomean':('gmpo_geomean_vs_sft.jsonl','gmpo_geomean_vs_sft.json','gmpo_geomean_vs_sft.independent.json',34,-29.296875,-27.982143140284066),
 'dhpo_average':('dhpo_average_vs_sft.jsonl','dhpo_average_vs_sft.json','dhpo_average_vs_sft.independent.json',46,-24.609375,-23.294643140284066),
}
verified={}
for method,(compact,report_file,independent_file,expected_success,expected_sr,expected_spl) in pairs.items():
 rel=f'runlogs/sequence_evaluation/{compact}'
 pair_rows=[json.loads(line) for line in (ROOT/rel).read_text().splitlines() if line]
 ids=[str(x['episode_id']) for x in pair_rows]
 assert len(pair_rows)==256 and len(set(ids))==256 and set(ids)==set(episode_ids)
 assert len({x['scene_id'] for x in pair_rows})==8
 for row in pair_rows:
  assert row['candidate_success'] in (0,1) and row['control_success'] in (0,1)
  assert math.isfinite(row['candidate_spl']) and 0<=row['candidate_spl']<=1
  assert math.isfinite(row['control_spl']) and 0<=row['control_spl']<=1
 candidate=sum(x['candidate_success'] for x in pair_rows)
 sft=sum(x['control_success'] for x in pair_rows)
 cspl=sum(x['candidate_spl'] for x in pair_rows)/256
 sspl=sum(x['control_spl'] for x in pair_rows)/256
 sr_pp=100*(candidate-sft)/256
 spl_pp=100*(cspl-sspl)
 item=load(f'runlogs/sequence_evaluation/{report_file}')
 recount=load(f'runlogs/sequence_evaluation/{independent_file}')
 assert candidate==expected_success and sft==109
 assert math.isclose(sr_pp,expected_sr,abs_tol=1e-10,rel_tol=0)
 assert math.isclose(spl_pp,expected_spl,abs_tol=1e-10,rel_tol=0)
 assert math.isclose(sr_pp,item['paired_sr_points'],abs_tol=1e-10,rel_tol=0)
 assert math.isclose(spl_pp,item['paired_spl_points'],abs_tol=1e-10,rel_tol=0)
 assert math.isclose(sr_pp,recount['paired_sr_points'],abs_tol=1e-10,rel_tol=0)
 assert math.isclose(spl_pp,recount['paired_spl_points'],abs_tol=1e-10,rel_tol=0)
 assert item['manifest_sha256']==manifest_sha and item['episodes']==256 and item['scenes']==8 and item['inference_errors']==0
 scene=report['comparisons'][method]
 assert scene['candidate_successes']==candidate
 assert math.isclose(scene['paired_sr_points'],sr_pp,abs_tol=1e-10,rel_tol=0)
 assert math.isclose(scene['paired_spl_points'],spl_pp,abs_tol=1e-10,rel_tol=0)
 assert report['gates'][method]['eligible_for_scale'] is False
 assert scene['scene_bootstrap_95pct_sr_points'][1]<0 and scene['scene_bootstrap_95pct_spl_points'][1]<0
 verified[method]={'candidate_successes':candidate,'sft_successes':sft,'paired_sr_points':sr_pp,'paired_spl_points':spl_pp}

for method in pairs:
 slug={'gspo_sequence':'seq_gspo_sequence_n4_seed11','gmpo_geomean':'seq_gmpo_geomean_n4_seed11','dhpo_average':'seq_dhpo_average_n4_seed11'}[method]
 for fname,steps in [('smoke_gradient_audit.json',2),('train_gradient_audit_64.json',64)]:
  audit=load(f'runlogs/{slug}/{fname}')
  assert audit['expected_steps']==steps and audit['observed_steps']==steps
  assert audit['nonzero_gradient_steps']==steps

print(json.dumps({'status':'PASS','manifest_episodes':256,'scenes':8,'raw_internal_ids_verified_remotely':1024,'local_compact_pairs_recounted':3,'inference_errors':0,'comparisons':verified,'all_methods_scale_eligible':False,'scope':'single configured seed on adaptively reused R2R-train development; not a clean generalization test'},indent=2))
