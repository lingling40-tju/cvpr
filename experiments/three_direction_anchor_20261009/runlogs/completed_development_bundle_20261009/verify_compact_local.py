#!/usr/bin/env python3
"""Independent compact recount for the frozen 3-method n=4 pilot."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SUITE=ROOT/'runlogs/development_suite'
def load(p): return json.loads((ROOT/p).read_text())
def sha(p): return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
manifest_rel='prepared_data/development256.json'
manifest=load(manifest_rel)
manifest_sha='8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3'
assert sha(manifest_rel)==manifest_sha
ids=[str(x) for x in manifest['episode_ids']]
assert len(ids)==len(set(ids))==256 and len(set(manifest['scene_ids']))==8
identity_sha='0ad45c7e518166fd8ffce502b7bd6354b201cce12729bd9e127e33ff9a319a99'
assert sha('runlogs/freeze/evaluation_identity.json')==identity_sha
aggregate=load('runlogs/development_suite/three_direction_report.json')
assert sha('runlogs/development_suite/three_direction_report.json')=='f16ac920e98be92a84eb500c483af3e89c4fbd3dcd86ff7d04f8e3285588cb63'
assert aggregate['evaluation_identity_sha256']==identity_sha
rawid=load('runlogs/development_suite/final_raw_id_independent_recount.json')
assert rawid['status']=='PASS' and rawid['raw_internal_ids_checked']==1024
assert rawid['episodes_per_model']==256 and rawid['models']==4 and rawid['inference_errors']==0
assert len(rawid['raw_stats_sha256'])==1024
assert rawid['aggregate_sha256']==sha('runlogs/development_suite/three_direction_report.json')
assert rawid['verifier_sha256']=='466cdc0717aec7042bf11cf975c5999ad36d6133f01a24656d88988bbf68020a'
assert (SUITE/'suite.completed').is_file()
assert (ROOT/'runlogs/final_raw_id_verification/watcher.completed').is_file()

keys=['grpo_anchor_vs_sft','turn_rloo_vs_sft','turn_rloo_vs_grpo','srgpo_vs_sft','srgpo_vs_grpo']
verified={}
for key in keys:
    rel=f'runlogs/development_suite/{key}.jsonl'
    compact=(ROOT/rel).read_bytes()
    assert hashlib.sha256(compact).hexdigest()==aggregate['pair_evidence_sha256'][f'{key}.jsonl']
    rows=[json.loads(line) for line in compact.decode().splitlines() if line]
    rowids=[str(r['episode_id']) for r in rows]
    assert len(rows)==256 and len(set(rowids))==256 and set(rowids)==set(ids)
    assert len({r['scene_id'] for r in rows})==8
    for r in rows:
        assert r['candidate_success'] in (0,1) and r['control_success'] in (0,1)
        for field in ('candidate_spl','control_spl'):
            x=r[field]; assert isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1
    report=load(f'runlogs/development_suite/{key}.json')
    recount=load(f'runlogs/development_suite/{key}.independent.json')
    c=sum(r['candidate_success'] for r in rows); b=sum(r['control_success'] for r in rows)
    cs=sum(r['candidate_spl'] for r in rows)/256; bs=sum(r['control_spl'] for r in rows)/256
    sr=100*(c-b)/256; spl=100*(cs-bs)
    assert c==report['candidate_successes'] and b==report['control_successes']
    assert math.isclose(sr,report['paired_sr_points'],abs_tol=1e-10,rel_tol=0)
    assert math.isclose(spl,report['paired_spl_points'],abs_tol=1e-10,rel_tol=0)
    assert math.isclose(sr,recount['paired_sr_points'],abs_tol=1e-10,rel_tol=0)
    assert math.isclose(spl,recount['paired_spl_points'],abs_tol=1e-10,rel_tol=0)
    assert report['episodes']==256 and report['scenes']==8 and report['inference_errors']==0
    assert report['manifest_sha256']==manifest_sha
    assert sha(rel)==aggregate['pair_evidence_sha256'][f'{key}.jsonl']
    verified[key]={'candidate_successes':c,'control_successes':b,'paired_sr_points':sr,'paired_spl_points':spl}

expected={
 'grpo_anchor_vs_sft':(-24.21875,-22.904018140284066),
 'turn_rloo_vs_sft':(-1.5625,-0.5300632274740147),
 'turn_rloo_vs_grpo':(22.65625,22.373954912810053),
 'srgpo_vs_sft':(-12.5,-11.316640681987277),
 'srgpo_vs_grpo':(11.71875,11.587377458296789),
}
for k,(sr,spl) in expected.items():
 assert math.isclose(verified[k]['paired_sr_points'],sr,abs_tol=1e-10,rel_tol=0)
 assert math.isclose(verified[k]['paired_spl_points'],spl,abs_tol=1e-10,rel_tol=0)

for method in ('grpo_anchor','turn_rloo','srgpo'):
    gate=aggregate['gates'][method]
    assert gate['eligible_for_scale'] is False and gate['pass_vs_sft'] is False
    name=f'{method}_vs_sft'
    assert verified[name]['paired_sr_points']<2 and verified[name]['paired_spl_points']<2
for method in ('turn_rloo','srgpo'):
    assert verified[f'{method}_vs_grpo']['paired_sr_points']>=2
    assert verified[f'{method}_vs_grpo']['paired_spl_points']>=2

training={}
for method,filename in [('grpo_anchor','grpo_anchor_train_audit.json'),('turn_rloo','turn_rloo_train_audit.json'),('srgpo','srgpo_train_audit.json')]:
    a=load(f'runlogs/development_suite/{filename}')
    assert a['optimizer_updates']==64 and a['nonzero_gradient_steps']==64
    assert a['reward_signal_steps']==64 and a['episode_exposures']==512
    assert a['recorded_trajectories']==2048 and a['group_size']==4
    training[method]={'updates':a['optimizer_updates'],'nonzero_gradient_steps':a['nonzero_gradient_steps'],'episode_exposures':a['episode_exposures'],'recorded_trajectories':a['recorded_trajectories']}

print(json.dumps({'status':'PASS','manifest_episodes':256,'scenes':8,'raw_internal_ids_verified_remotely':1024,'local_compact_pairs_recounted':len(verified),'inference_errors':0,'training':training,'comparisons':verified,'all_methods_scale_eligible':False,'scope':'one configured seed on adaptively reused R2R-train development scenes; exploratory, not clean generalization'},indent=2))
