"""Small source-integrity checks; requires the remote training environment."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from train_future_advantage_sparse_lora import digest, forward_meters, load_data


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(root: Path) -> tuple[Path, Path, Path]:
    replay = root / "replay"
    selected = {}
    pairs = {}
    counts = {}
    for part, scene in (("fit", "scene_a"), ("development", "scene_b")):
        selected[part] = [{"record_id": f"s11_e1_v{variant}", "seed": 11,
                           "episode_id": "1", "variant": variant,
                           "scene_id": scene, "terminal_mode": "timeout"}
                          for variant in (0, 1)]
        pairs[part] = [{"pair_id": f"{part}_{anchor}", "seed": 11,
                        "episode_id": "1", "scene_id": scene,
                        "anchor": anchor, "left_variant": 0,
                        "right_variant": 1, "preferred_variant": 0,
                        "future_return_gap_for_label_only": .5,
                        "same_terminal_mode": True}
                       for anchor in (3, 6)]
        counts[part] = {str(anchor): {
            "pairs": 1, "unique_episode_groups": 1,
            "same_terminal_mode_pairs": 1,
            "same_terminal_mode_unique_episode_groups": 1}
            for anchor in (3, 6)}
    manifest = {"schema": "future_advantage_sparse_replay_manifest_v1",
                "group_size": 4, "anchors": [3, 6], "seeds": [11],
                "preflight_report_sha256": "fixed_gate_hash",
                "selected": selected, "counts": counts}
    manifest_path = root / "manifest.json"
    write(manifest_path, manifest)
    for part in selected:
        for row in selected[part]:
            write(replay / part / "records" / f"{row['record_id']}.json", {
                "schema": "future_advantage_sparse_model_input_v1",
                "record_id": row["record_id"],
                "manifest_sha256": digest(manifest_path),
                "input": {"instruction": "Walk toward the chair.",
                          "images": {}, "action_history_by_anchor": {}}})
    labels = {"schema": "future_advantage_within_group_pair_labels_v1",
              "group_size": 4, "anchors": [3, 6], "seeds": [11],
              "preflight_report_sha256": "fixed_gate_hash",
              "replay_manifest_sha256": digest(manifest_path),
              "pairs": pairs, "counts": counts}
    labels_path = root / "labels.json"
    write(labels_path, labels)
    return manifest_path, labels_path, replay


def rejects(call, message: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="future_advantage_integrity_") as tmp:
        manifest, labels, replay = fixture(Path(tmp))
        records, pairs, sources = load_data(manifest, labels, replay)
        assert len(records["fit"]) == len(records["development"]) == 2
        assert len(pairs["fit"]) == len(pairs["development"]) == 2
        assert sources["manifest"] == digest(manifest)
        changed = json.loads(labels.read_text())
        changed["pairs"]["fit"][0]["scene_id"] = "wrong_scene"
        write(labels, changed)
        rejects(lambda: load_data(manifest, labels, replay),
                "mismatched scene reached the model")
        changed["pairs"]["fit"][0]["scene_id"] = "scene_a"
        changed["pairs"]["fit"][0]["same_terminal_mode"] = False
        write(labels, changed)
        rejects(lambda: load_data(manifest, labels, replay),
                "false same-terminal-mode supervision accepted")
        record = {"input": {"action_history_by_anchor": {"3": [
            {"executed_actions": ["turn left 15deg", "move forward 25cm"]}]}}}
        assert forward_meters(record, 3) == .25
        record["input"]["action_history_by_anchor"]["3"][0][
            "executed_actions"][1] = "move forward 0.25m"
        rejects(lambda: forward_meters(record, 3),
                "unknown forward-action unit silently changed baseline")
    print("source integrity checks passed")


if __name__ == "__main__":
    main()
