"""Fail-closed, pre-completion recount of every frozen GAE eval episode.

This runs before the suite completion marker is written. The separate
verify_gae_raw.py reruns the raw recount after the marker and must agree.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


MANIFEST_SHA256 = "03c74d3e9b57da28895289de57b317f85eff3108ce806a7375865a3f654d24ed"
CONTROL = "qwen3_exact_control_64step_seed11"
CANDIDATE = "turn_gae_64step_seed11"
ARMS = (CONTROL, CANDIDATE)


def metric(raw: dict, episode_id: str, label: str) -> tuple[int, float]:
    declared = raw.get("episode_id")
    if declared is not None and str(declared) != episode_id:
        raise ValueError(f"episode ID mismatch: {label} {episode_id}")
    if raw.get("early_stop_reason") == "inference_error":
        raise ValueError(f"inference error: {label} {episode_id}")
    success = raw.get("success")
    if isinstance(success, bool):
        success = int(success)
    if success not in (0, 0.0, 1, 1.0):
        raise ValueError(f"invalid success: {label} {episode_id}")
    spl = raw.get("spl")
    if isinstance(spl, bool) or not isinstance(spl, (int, float)) or not math.isfinite(spl) or not 0 <= spl <= 1:
        raise ValueError(f"invalid SPL: {label} {episode_id}")
    return int(success), float(spl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_root", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    root = args.result_root
    if (root / "suite.completed").exists() or (root / "suite.failed").exists():
        raise ValueError("preverification requires an unmarked, healthy suite")
    manifest_bytes = args.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("frozen manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    ids = [str(x) for x in manifest["episode_ids"]]
    scenes = [str(x) for x in manifest["scene_ids"]]
    if manifest["split"] != "val_seen" or len(ids) != 778 or len(set(ids)) != 778 or len(scenes) != 778 or len(set(scenes)) != 53:
        raise ValueError("unexpected manifest coverage")

    validators = {}
    for label in ARMS:
        if not (root / f"{label}.completed").exists() or (root / f"{label}.failed").exists():
            raise ValueError(f"arm incomplete or failed: {label}")
        validator = json.loads((root / f"{label}.validated.json").read_text())
        if validator.get("label") != label or validator.get("episodes") != 778 or validator.get("inference_errors") != 0:
            raise ValueError(f"invalid validator: {label}")
        validators[label] = validator
        for shard in range(4):
            folder = root / label / f"shard_{shard:02d}" / "log"
            expected = {f"stats_{ids[i]}_0.json" for i in range(shard, 778, 4)}
            observed = {path.name for path in folder.glob("stats_*_0.json")}
            if observed != expected:
                raise ValueError(f"shard coverage mismatch: {label} {shard}")

    successes = {label: 0 for label in ARMS}
    spl_sums = {label: 0.0 for label in ARMS}
    candidate_only = control_only = 0
    raw_hash = hashlib.sha256()
    for index, episode_id in enumerate(ids):
        values = {}
        for label in ARMS:
            path = root / label / f"shard_{index % 4:02d}" / "log" / f"stats_{episode_id}_0.json"
            content = path.read_bytes()
            raw_hash.update(path.relative_to(root).as_posix().encode() + b"\0" + content)
            values[label] = metric(json.loads(content), episode_id, label)
            successes[label] += values[label][0]
            spl_sums[label] += values[label][1]
        candidate_only += int(values[CANDIDATE][0] == 1 and values[CONTROL][0] == 0)
        control_only += int(values[CANDIDATE][0] == 0 and values[CONTROL][0] == 1)
    for label in ARMS:
        if successes[label] != validators[label].get("successes"):
            raise ValueError(f"raw and validator success counts differ: {label}")
    if candidate_only - control_only != successes[CANDIDATE] - successes[CONTROL]:
        raise ValueError("paired discordance mismatch")
    sr_points = 100 * (successes[CANDIDATE] - successes[CONTROL]) / 778
    spl_points = 100 * (spl_sums[CANDIDATE] - spl_sums[CONTROL]) / 778
    result = {
        "schema": "gae_val_seen778_precompletion_raw_recount_v1",
        "manifest_sha256": MANIFEST_SHA256,
        "raw_stats_sha256": raw_hash.hexdigest(),
        "episodes": 778,
        "scenes": 53,
        "control_successes": successes[CONTROL],
        "candidate_successes": successes[CANDIDATE],
        "control_spl": spl_sums[CONTROL] / 778,
        "candidate_spl": spl_sums[CANDIDATE] / 778,
        "paired_sr_points": sr_points,
        "paired_spl_points": spl_points,
        "candidate_only_success": candidate_only,
        "control_only_success": control_only,
        "inference_errors": 0,
        "advance_gate_passed": sr_points >= 2.0 and spl_points >= 2.0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
