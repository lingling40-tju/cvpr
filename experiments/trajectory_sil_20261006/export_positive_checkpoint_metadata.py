#!/usr/bin/env python3
"""Record CPU header/extent checks for an already completed 128-step export.

This reads metadata, not tensor payloads. It does not perform inference,
change completion markers, or overwrite an existing report.
"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("control", "candidate"))
    parser.add_argument("--seed", required=True, type=int, choices=(11, 22, 33))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    label = "positive_trajectory_{}_128step_seed{}".format(args.arm, args.seed)
    run = args.root / "runlogs" / label
    require((run / "completed").is_file() and not (run / "failed").exists(),
            "Training is not successfully completed")
    completed = (run / "completed").read_text().strip()
    require(bool(completed), "Empty completion marker")
    audit_path = args.root / "runlogs/positive_scale" / (
        "{}_seed{}_train_audit.json".format(args.arm, args.seed))
    audit = read_json(audit_path)
    require(audit["schema"] == "positive_trajectory_training_audit_v1" and
            audit["arm"] == args.arm and
            audit["expected_steps"] == audit["observed_steps"] == 128,
            "Training audit identity or coverage differs")
    require(audit["positive_advantage_steps"] > 0 and
            audit["nonzero_actor_gradient_steps"] > 0 and
            audit["kl_loss_metric_present_and_finite"] is True,
            "Training audit has no finite actor signal")
    require(audit["train_log_sha256"] == sha256(run / "train.log"),
            "Audited training log changed")

    config_path = run / "config.txt"
    config_fields = unique_object(item.split("=", 1) for item in
                                  config_path.read_text().split())
    expected_config = {
        "arm": args.arm, "seed": str(args.seed), "steps": "128",
        "rows": "512", "batch_rows": "8", "group_n": "4",
        "dataset_sha256": "240a0eea6756a78b50586ac29ec615827bf67238674c19423a16ff2e9654256e",
        "reward": "weighted_success15_plus_ndtw5",
        "loss_agg": "seq-mean-token-mean", "kl_loss_coef": "0.01",
    }
    require(all(config_fields.get(key) == value for key, value in
                expected_config.items()), "Frozen run configuration differs")

    hf = args.root / "verl_checkpoints" / label / "global_step_128/actor/huggingface"
    index_path = hf / "model.safetensors.index.json"
    model_config_path = hf / "config.json"
    index = read_json(index_path)
    model_config = read_json(model_config_path)
    weights = index["weight_map"]
    names = sorted(set(weights.values()))
    require(len(weights) == 825 and len(names) == 4,
            "Frozen model tensor or shard count differs")
    require(all(isinstance(name, str) and Path(name).name == name and
                name.endswith(".safetensors") for name in names),
            "Invalid shard filename")
    require({path.name for path in hf.glob("*.safetensors")} == set(names),
            "Physical shard set differs from index")
    require(model_config["architectures"] == ["Qwen2_5_VLForConditionalGeneration"] and
            model_config["torch_dtype"] == "float32", "Frozen model configuration differs")

    shards = {}
    seen = set()
    payload_bytes = 0
    for name in names:
        path = hf / name
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            require(len(prefix) == 8, "Missing safetensors length prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            require(0 < header_size <= 10_000_000 and 8 + header_size <= size,
                    "Invalid safetensors header size")
            header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        require(set(entries) == {key for key, value in weights.items() if value == name},
                "Tensor names differ from index")
        require(not seen.intersection(entries), "Tensor duplicated across shards")
        seen.update(entries)
        intervals = []
        for entry in entries.values():
            require(entry["dtype"] == "F32", "Unexpected tensor dtype")
            shape, offsets = entry["shape"], entry["data_offsets"]
            require(all(type(dim) is int and dim >= 0 for dim in shape), "Invalid shape")
            require(len(offsets) == 2 and all(type(x) is int and x >= 0 for x in offsets),
                    "Invalid payload offsets")
            start, end = offsets
            require(end - start == math.prod(shape) * 4, "Shape and payload extent differ")
            intervals.append((start, end))
        cursor = 0
        for start, end in sorted(intervals):
            require(start == cursor, "Payload ranges overlap or contain a gap")
            cursor = end
        require(8 + header_size + cursor == size, "File extent differs from header")
        payload_bytes += cursor
        shards[name] = {
            "bytes": size, "tensors": len(entries), "dtypes": ["F32"],
            "header_sha256": hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
        }
    require(seen == set(weights), "Export tensor coverage differs")
    require(index["metadata"]["total_size"] == payload_bytes,
            "Index and shard payload byte totals differ")
    require((run / "completed").read_text().strip() == completed and
            not (run / "failed").exists(), "Completion state changed during check")

    report = {
        "schema": "trained_checkpoint_metadata_extent_check_v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": label, "completed": completed, "shards": shards, "tensors": len(seen),
        "index_sha256": sha256(index_path), "config_sha256": sha256(model_config_path),
        "architectures": model_config["architectures"], "dtype": model_config["torch_dtype"],
        "header_hash_encoding": "SHA-256 of json.dumps(header, sort_keys=True).encode()",
        "scope": "CPU metadata and exact file extent check; not model inference or tensor-value validation",
        "training_audit_sha256": sha256(audit_path),
        "train_log_sha256": audit["train_log_sha256"],
        "training_config_sha256": sha256(config_path), "exporter_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("label", "completed", "tensors", "scope")}))


if __name__ == "__main__":
    main()
