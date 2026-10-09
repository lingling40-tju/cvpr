"""No-GPU verification of the fixed evaluation sources and reused SFT raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--identity-sha', required=True)
    a = p.parse_args()
    path = a.root / 'runlogs/freeze/evaluation_identity.json'
    if sha(path) != a.identity_sha:
        raise ValueError('evaluation identity differs from frozen launch')
    x = json.loads(path.read_text())
    if x['schema'] != 'three_direction_development_eval_v1' or x['episodes'] != 256 or x['group_size'] != 4:
        raise ValueError('evaluation identity schema/budget differs')
    if sha(a.root / 'runlogs/freeze/identity.json') != x['training_identity_sha256']:
        raise ValueError('parent training identity changed')
    parent = json.loads((a.root / 'runlogs/freeze/identity.json').read_text())
    for rel, digest in parent['source_sha256'].items():
        if sha(a.root / rel) != digest:
            raise ValueError('parent training source changed: ' + rel)
    for rel, digest in x['source_sha256'].items():
        if sha(a.root / rel) != digest:
            raise ValueError('evaluation source changed: ' + rel)
    for name, digest in x['sft_reference']['evidence_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('reused SFT evidence changed: ' + name)
    print(json.dumps({'status': 'PASS', 'evaluation_identity_sha256': a.identity_sha,
                      'sources': len(x['source_sha256']), 'sft_evidence_files': len(x['sft_reference']['evidence_sha256'])}))


if __name__ == '__main__':
    main()
