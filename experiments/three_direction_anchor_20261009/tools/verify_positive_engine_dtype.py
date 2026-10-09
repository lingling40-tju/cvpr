"""Read actual vLLM engine startup configuration; no model or GPU calls."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_engine(log, checkpoint, expected_dtype):
    lines = [line for line in log.read_text(errors="replace").splitlines()
             if "Initializing a V" in line and "LLM engine" in line]
    if len(lines) != 1:
        raise ValueError(f"expected one engine startup: {log}, got {len(lines)}")
    line = lines[0]
    model = re.search(r"\bmodel=['\"]([^'\"]+)['\"]", line)
    dtype = re.search(r"\bdtype=torch\.([a-z0-9_]+)\b", line)
    if model is None or model.group(1) != str(checkpoint):
        raise ValueError(f"engine checkpoint differs: {log}")
    if dtype is None or dtype.group(1) != expected_dtype:
        raise ValueError(f"engine dtype differs: {log}")
    for key, expected in (("max_seq_len", "16384"), ("seed", "11"),
                          ("tensor_parallel_size", "1"), ("pipeline_parallel_size", "1"),
                          ("quantization", "None"), ("kv_cache_dtype", "auto")):
        value = re.search(r"\b" + key + r"=([^,\s]+)", line)
        if value is None or value.group(1) != expected:
            raise ValueError(f"engine {key} differs: {log}")
    config = checkpoint / "config.json"
    return {"schema": "positive_actual_engine_dtype_v1", "log": str(log),
            "log_snapshot_sha256": sha(log), "checkpoint": str(checkpoint),
            "checkpoint_config_sha256": sha(config),
            "checkpoint_config_dtype": json.loads(config.read_text())["torch_dtype"],
            "actual_engine_dtype": dtype.group(1), "engine_seed": 11,
            "max_seq_len": 16384, "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "quantization": None, "kv_cache_dtype": "auto",
            "engine_startup_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "verifier_sha256": sha(Path(__file__)),
            "scope": "Actual startup log identity, not weight-value or navigation validation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-dtype", choices=("float16", "bfloat16"), required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = inspect_engine(a.log, a.checkpoint, a.expected_dtype)
    if a.output is not None:
        if a.output.exists():
            old = json.loads(a.output.read_text())
            old.pop("verified_at_utc", None)
            if old != report:
                raise ValueError(f"existing proof differs; preserve it: {a.output}")
        else:
            report["verified_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with a.output.open("x") as handle:
                handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
