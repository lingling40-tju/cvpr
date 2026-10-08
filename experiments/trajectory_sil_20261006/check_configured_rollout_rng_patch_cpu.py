#!/usr/bin/env python3
"""Check the unapplied manager RNG patch with a CPU generator and real source."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


ORIGINAL_SHA = "e1204549388f04853e89a8baffc608c27b9fa4fd5cfa95eb332fb67e822e637d"
PATCHED_SHA = "f1a6c9d0ee5315a64fd0178cf039266ce00eea5ef107f8991aa2ae46e72fe9e0"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def apply_one_hunk(source, patch):
    lines = patch.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("@@ ")]
    if len(starts) != 1:
        raise ValueError("Expected the frozen single-hunk patch")
    hunk = lines[starts[0] + 1:]
    if any(line[:1] not in (" ", "+", "-") for line in hunk):
        raise ValueError("Unexpected patch line")
    old = "".join(line[1:] for line in hunk if line[:1] in (" ", "-"))
    new = "".join(line[1:] for line in hunk if line[:1] in (" ", "+"))
    if source.count(old) != 1:
        raise ValueError("Frozen patch context does not match exactly once")
    return source.replace(old, new, 1)


def constructor_rng_subset(source):
    tree = ast.parse(source)
    constructors = [node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    starts = []
    for constructor in constructors:
        for i, node in enumerate(constructor.body):
            if (isinstance(node, ast.Assign) and
                    ast.get_source_segment(source, node).startswith("self.torch_random_states =")):
                starts.append(constructor.body[i:i + 2])
    if len(starts) != 1 or not isinstance(starts[0][1], ast.If):
        raise ValueError("Could not isolate the actual RNG constructor statements")
    return compile(ast.Module(body=starts[0], type_ignores=[]), "<manager_rng_subset>", "exec")


class CPUDevice:
    def __init__(self):
        self.generator = torch.Generator(device="cpu").manual_seed(777)

    def manual_seed(self, seed):
        self.generator.manual_seed(seed)

    def get_rng_state(self):
        return self.generator.get_state().clone()

    def set_rng_state(self, state):
        self.generator.set_state(state)


def run_case(code, seed, rank):
    device = CPUDevice()
    training_state = device.get_rng_state()
    manager = SimpleNamespace(rollout_config={"seed": seed},
                              device_mesh={"dp": SimpleNamespace(get_local_rank=lambda: rank)})
    exec(code, {"self": manager, "get_torch_device": lambda: device})
    if not torch.equal(device.get_rng_state(), training_state):
        raise ValueError("Training generator state was not restored")
    return manager.gen_random_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    source_bytes = args.source.read_bytes()
    if digest(source_bytes) != ORIGINAL_SHA:
        raise ValueError("Original source identity differs")
    patch_bytes = args.patch.read_bytes()
    patched = apply_one_hunk(source_bytes.decode(), patch_bytes.decode())
    if digest(patched.encode()) != PATCHED_SHA:
        raise ValueError("Patched source identity differs")
    old_code = constructor_rng_subset(source_bytes.decode())
    new_code = constructor_rng_subset(patched)
    old11, old22 = run_case(old_code, 11, 0), run_case(old_code, 22, 0)
    new11, new22 = run_case(new_code, 11, 0), run_case(new_code, 22, 0)
    checks = {
        "legacy_common_state_for_configured_seeds_11_22": torch.equal(old11, old22),
        "patched_distinct_states_for_configured_seeds_11_22": not torch.equal(new11, new22),
        "same_configured_seed_and_dp_rank_repeat_same_state": torch.equal(new11, run_case(new_code, 11, 0)),
        "different_dp_ranks_have_distinct_states": not torch.equal(new11, run_case(new_code, 11, 1)),
        "training_state_restored_in_all_successful_cases": True,
    }
    try:
        run_case(new_code, None, 0)
    except ValueError as error:
        checks["missing_configured_seed_rejected"] = str(error) == "Configured rollout.seed is required for seeded sampling"
    else:
        checks["missing_configured_seed_rejected"] = False
    if not all(checks.values()):
        raise ValueError("CPU generator fixture failed: {}".format(checks))
    report = {"schema": "configured_rollout_rng_patch_cpu_check_v1",
              "original_source_sha256": ORIGINAL_SHA, "patched_source_sha256": PATCHED_SHA,
              **checks, "patch_sha256": digest(patch_bytes),
              "fixture_sha256": digest(Path(__file__).read_bytes()),
              "torch_version": torch.__version__,
              "scope": "CPU torch.Generator fixture executing the RNG constructor subset; no live CUDA worker, VLM inference, or navigation result"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
