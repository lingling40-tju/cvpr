#!/usr/bin/env python3
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]); identity_path=Path(sys.argv[2]); sha_path=Path(sys.argv[3]); runner=Path(sys.argv[4])
state=root/"runlogs/remax_reserved_eval"; freeze=root/"runlogs/freeze"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
ident=json.loads(identity_path.read_text())
assert sha(identity_path)==sha_path.read_text().strip()
assert ident["schema"]=="remax_reserved_eval_freeze_v1"
assert ident["training_identity_sha256"]=="d7a218037ea5e8ca7ad775df64ebcd9fdb22dcfd48af1f32e340e244faec207f"
assert sha(freeze/"identity.json")==ident["training_identity_sha256"]
train_identity=json.loads((freeze/"identity.json").read_text())
assert sha(freeze/"protocol.txt")==ident["protocol_sha256"]
assert sha(freeze/"source_files.sha256")==ident["training_source_manifest_sha256"]
assert sha(state/"reserved_eval_protocol.txt")==ident["evaluation_protocol_sha256"]
assert sha(root/"prepared_data/reserved256.json")==ident["reserved_manifest_sha256"]
assert sha(runner)==ident["runner_sha256"]
assert sha(state/"verify_remax_eval_freeze.py")==ident["verifier_sha256"]
for rel,digest in ident["evaluation_source_sha256"].items():
    assert sha(root/rel)==digest,(rel,sha(root/rel),digest)
sft=Path(ident["sft_evidence_root"])
assert sha(sft/"positive_initial_sft.validated.json")==ident["sft_validator_sha256"]
assert sha(root/"prepared_data/positive_initial_sft_identity.json")==ident["sft_checkpoint_identity_sha256"]
sft_identity=json.loads((root/"prepared_data/positive_initial_sft_identity.json").read_text())
assert ident["sft_checkpoint_identity_sha256"]==train_identity["sft_identity_sha256"]
assert sft_identity["rl_optimizer_steps"]==0
sftval=json.loads((sft/"positive_initial_sft.validated.json").read_text())
assert sftval["episodes"]==256 and sftval["successes"]==91 and sftval["inference_errors"]==0
assert sftval["manifest_sha256"]==ident["reserved_manifest_sha256"]
assert sftval["scenes"]==8 and sftval["raw_episode_id_checked"] is True
assert math.isclose(sftval["spl"],ident["sft_baseline"]["spl"],rel_tol=0,abs_tol=1e-12)
assert ident["sft_validator_sha256"]==train_identity["reserved_sft_validator_sha256"]
sftdtype=json.loads((sft/"positive_initial_sft.dtype.json").read_text())
assert sftdtype["actual_engine_dtype"]=="float16" and sftdtype["engine_seed"]==11
assert sha(sft/"positive_initial_sft.dtype.json")==ident["sft_dtype_sha256"]
assert sha(sft/"vllm_positive_initial_sft.log")==ident["sft_engine_log_sha256"]
raw_manifest=freeze/"sft_reserved_raw_stats.sha256"
assert sha(raw_manifest)==ident["sft_raw_manifest_sha256"]
assert len(raw_manifest.read_text().splitlines())==256
assert ident["sft_baseline"]["successes"]==91 and ident["sft_baseline"]["episodes"]==256
assert math.isclose(ident["sft_baseline"]["spl"],0.3443758508038736,rel_tol=0,abs_tol=1e-12)
assert ident["evaluation"]["dtype"]=="float16" and ident["evaluation"]["seed"]==11
assert ident["evaluation"]["temperature"]==0.2 and ident["evaluation"]["top_p"]==0.8
assert ident["evaluation"]["max_tokens"]==512 and ident["evaluation"]["max_turns"]==12
assert ident["gate"]["paired_sr_and_spl_each_pp"]==2.0
print(json.dumps({"status":"PASS","identity_sha256":sha(identity_path),"runner_sha256":ident["runner_sha256"],"source_files":len(ident["evaluation_source_sha256"]),"sft_raw_stats":256}))

