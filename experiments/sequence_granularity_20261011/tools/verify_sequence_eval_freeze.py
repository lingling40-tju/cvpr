"""Verify the frozen sequence-method development evaluation identity."""
import argparse, hashlib, json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    a=ap.parse_args()
    root=a.root
    state=root/"runlogs/sequence_evaluation"
    identity=state/"evaluation_identity.json"
    recorded=(state/"evaluation_identity.sha256").read_text().split()[0]
    assert sha(identity)==recorded
    d=json.loads(identity.read_text())
    assert d["schema"]=="sequence_granularity_evaluation_v1"
    assert sha(root/"prepared_data/development256.json")==d["manifest_sha256"]
    assert sha(root/"runlogs/sequence_training/freeze_identity.json")==d["training_identity_sha256"]
    for rel,want in d["source_sha256"].items():
        assert sha(root/rel)==want,rel
    for rel,want in d["custom_source_sha256"].items():
        assert sha(root/rel)==want,rel
    ref=state/"sft_reference_manifest.json"
    assert sha(ref)==d["sft_reference_manifest_sha256"]
    r=json.loads(ref.read_text())
    assert len(r["files"])==256 and r["successes"]==109 and r["inference_errors"]==0
    print(json.dumps({"status":"PASS","identity_sha256":recorded,"sources":len(d["source_sha256"])+len(d["custom_source_sha256"]),"sft_episode_files":len(r["files"])}))


if __name__=="__main__":
    main()
