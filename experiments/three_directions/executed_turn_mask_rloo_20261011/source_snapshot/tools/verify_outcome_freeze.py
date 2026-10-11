"""Verify the immutable follow-up source/data/reference identity."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--identity-sha', required=True)
    args = parser.parse_args()
    path = args.root / 'runlogs/freeze/identity.json'
    identity = json.loads(path.read_text())
    payload = {key: value for key, value in identity.items() if key != 'identity_sha256'}
    if canonical_sha(payload) != args.identity_sha or identity['identity_sha256'] != args.identity_sha:
        raise ValueError('freeze identity SHA differs')
    for rel, digest_value in identity['source_sha256'].items():
        if digest(args.root / rel) != digest_value:
            raise ValueError('source changed: ' + rel)
    for label, files in identity['references'].items():
        for file_path, digest_value in files.items():
            if digest(file_path) != digest_value:
                raise ValueError(f'{label} comparator/reference evidence changed: {file_path}')
    print(json.dumps({'status': 'PASS', 'identity_sha256': args.identity_sha,
                      'source_count': len(identity['source_sha256']),
                      'reference_file_counts': {k: len(v) for k, v in identity['references'].items()}}))


if __name__ == '__main__':
    main()
