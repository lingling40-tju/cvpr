import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); x=json.loads((r/"runlogs/freeze/identity.json").read_text())
assert x["schema"]=="three_direction_n4_sft_anchored_pilot_v1"
assert x["common"]["group_size"]==4 and x["common"]["steps"]==64 and x["common"]["configured_seed"]==11
assert x["primary_gate"]["scale"].startswith("if a candidate clears")
for n,h in x["source_sha256"].items():
 q=r/n
 if q.is_symlink(): digest=hashlib.sha256(q.read_bytes()).hexdigest()
 else: digest=hashlib.sha256(q.read_bytes()).hexdigest()
 assert digest==h,(n,digest,h)
print(json.dumps({"freeze_sha256":hashlib.sha256((r/"runlogs/freeze/identity.json").read_bytes()).hexdigest(),"frozen_sources":len(x["source_sha256"]),"n":4,"steps":64,"status":"PASS"}))
