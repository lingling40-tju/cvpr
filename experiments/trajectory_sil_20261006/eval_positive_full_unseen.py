"""Complete frozen val-unseen inference using the pilot's request/resize shim."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import queue

from eval_train_scene_subset import habitat, get_config, evaluate_agent


MANIFEST_SHA = "262fcb8102bab3fb309e5f9f25a6527fdec5c9ae2ea87b12168e7f3cb24a538e"
SOURCE_SHA = "1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3"
GT_SHA = "46a2e02c4a5e3b9d3d3c936009cbe1a4e9ea8f34156589f1e69ff1ff7080c044"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-label", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--base-url", required=True)
    args = p.parse_args()
    assert args.shard_count == 4 and 0 <= args.shard_index < 4
    assert args.max_turns == 12
    assert hashlib.sha256(args.manifest.read_bytes()).hexdigest() == MANIFEST_SHA
    manifest = json.loads(args.manifest.read_text())
    ids = [str(x) for x in manifest["episode_ids"]]
    assert manifest["split"] == "val_unseen" and len(ids) == len(set(ids)) == 1839
    config = get_config("vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml")
    assert config.TASK_CONFIG.DATASET.SPLIT == config.EVAL.SPLIT == \
        config.TASK_CONFIG.TASK.NDTW.SPLIT == "val_unseen"
    source = Path(config.TASK_CONFIG.DATASET.DATA_PATH.format(split="val_unseen"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    gt = Path(config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split="val_unseen"))
    assert hashlib.sha256(gt.read_bytes()).hexdigest() == GT_SHA
    with gzip.open(gt, "rt") as f:
        references = json.load(f)
    assert all(references.get(i, {}).get("locations") for i in ids)
    dataset = habitat.datasets.make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE,
                                           config=config.TASK_CONFIG.DATASET)
    by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    assert set(by_id) == set(ids)
    all_episodes = [by_id[i] for i in ids]
    scenes = [str(ep.scene_id) for ep in all_episodes]
    assert scenes == manifest["scene_ids"] and len(set(scenes)) == 11
    chosen = all_episodes[args.shard_index::4]
    dataset.episodes = chosen
    if args.validate_only:
        print(json.dumps({"split": "val_unseen", "manifest_count": 1839,
                          "scene_count": 11, "shard_count": len(chosen),
                          "ndtw_split": "val_unseen", "missing_ndtw_references": 0,
                          "manifest_sha256": MANIFEST_SHA, "source_sha256": SOURCE_SHA,
                          "ndtw_reference_sha256": GT_SHA, "model_calls": 0}))
        return
    folder = args.result_root / args.model_label / f"shard_{args.shard_index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = args.base_url
    q = queue.Queue()
    for retry in range(3):
        evaluate_agent(q, "r2r", "EMPTY", args.base_url, config, dataset,
                       str(folder), 1, 76800, 12)
        rows, invalid = [], []
        for ep in chosen:
            path = folder / "log" / f"stats_{ep.episode_id}_0.json"
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                invalid.append(path)
                continue
            if str(row.get("id")) != str(ep.episode_id) or \
                    row.get("early_stop_reason") == "inference_error":
                invalid.append(path)
            else:
                rows.append(row)
        if not invalid:
            break
        if retry == 2:
            raise RuntimeError(f"{len(invalid)} invalid episodes in shard {args.shard_index}")
        for path in invalid:
            if path.exists():
                path.rename(path.with_name(f"{path.name}.failed_retry{retry + 1}_pid{os.getpid()}"))
    assert len(rows) == len(chosen)
    summary = {"label": args.model_label, "split": "val_unseen", "count": len(rows),
               "successes": sum(bool(r["success"]) for r in rows),
               "sr": sum(bool(r["success"]) for r in rows) / len(rows),
               "spl": sum(float(r["spl"]) for r in rows) / len(rows),
               "inference_errors": 0, "max_turns": 12,
               "episode_ids": [str(ep.episode_id) for ep in chosen],
               "shard_count": 4, "shard_index": args.shard_index}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
