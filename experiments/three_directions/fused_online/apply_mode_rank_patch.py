"""Install the checksum-guarded group-four ordinal reward in an isolated tree."""

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = "c4ac1e40651f45ccd4d4fc46b26ea0ff2ad9b5e1c88a3e5e0335d9b74f26cdc9"
IMPORT_ANCHOR = "from collections import defaultdict\n"
BODY_ANCHOR = "    reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]\n"
INJECT = '''    if config.agent.reward.get("mode_stratified_ordinal", False):
        removed, ordinal, mode_summary = mode_stratified_adjustments(
            info_list, sampling_params.n)
        for index, (old_bonus, new_ordinal) in enumerate(zip(removed, ordinal)):
            assert info_list[index]["done"]
            difference = new_ordinal - old_bonus
            reward_tensor_list[index][-1] += difference
            info_list[index]["total_reward"] += difference
            info_list[index]["reward_components"]["fused_bonus"] = 0.0
            info_list[index]["reward_components"]["mode_ordinal"] = new_ordinal
            info_list[index]["fused_reward"]["removed_bonus"] = old_bonus
            info_list[index]["fused_reward"]["applied_ordinal"] = new_ordinal
            info_list[index]["gen_traj"][-1]["reward"] += difference
            if abs(float(reward_tensor_list[index].sum()) -
                   float(info_list[index]["total_reward"])) > 1e-3:
                raise AssertionError("ordinal reward tensor/info mismatch")
        print(f"[mode stratified ordinal] {mode_summary}", flush=True)

'''


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("helper", type=Path)
    args = parser.parse_args()
    path = args.root / "verl/workers/agent/parallel_env_vlnce.py"
    original = path.read_text()
    if digest(path) != EXPECTED_SHA256:
        raise ValueError("ActiveVLN source hash mismatch or patch already applied")
    if original.count(IMPORT_ANCHOR) != 1 or original.count(BODY_ANCHOR) != 1:
        raise ValueError("patch anchor mismatch")
    patched = original.replace(IMPORT_ANCHOR,
                               IMPORT_ANCHOR +
                               "from .mode_stratified_reward import mode_stratified_adjustments\n")
    patched = patched.replace(BODY_ANCHOR, INJECT + BODY_ANCHOR)
    target = path.with_suffix(".py.mode_rank_source")
    target.write_text(original)
    path.write_text(patched)
    helper_target = path.with_name("mode_stratified_reward.py")
    helper_target.write_bytes(args.helper.read_bytes())
    print("source_sha256", EXPECTED_SHA256)
    print("patched_sha256", digest(path))
    print("helper_sha256", digest(helper_target))


if __name__ == "__main__":
    main()
