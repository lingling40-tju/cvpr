import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--identity',type=Path,required=True); p.add_argument('--identity-sha',required=True); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
assert sha(a.identity)==a.identity_sha
x=json.loads(a.identity.read_text()); assert x['schema']=='prefix_bc_grpo_v4_development_eval_v1'
for name,digest in x['source_sha256'].items():
 path=Path(name) if name.startswith('/') else a.root/name
 assert sha(path)==digest,(name,sha(path),digest)
for name,digest in x['candidate_checkpoint_files'].items():
 path=a.root/'verl_checkpoints/prefix_bc_grpo_n4_seed11_v4/global_step_64/actor/huggingface'/name
 assert sha(path)==digest,(name,sha(path),digest)
print(json.dumps({'status':'PASS','identity_sha256':a.identity_sha,'frozen_files':len(x['source_sha256']),'candidate_checkpoint_files':len(x['candidate_checkpoint_files'])}))
