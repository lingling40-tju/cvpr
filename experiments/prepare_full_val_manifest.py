"""Write the exact full R2R val-unseen episode manifest before sharded evaluation."""

import argparse
import json
from pathlib import Path

import habitat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from VLN_CE.vlnce_baselines.config.default import get_config

    config = get_config("vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml")
    assert config.TASK_CONFIG.DATASET.SPLIT == "val_unseen"
    dataset = habitat.datasets.make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    manifest = {
        "split": "val_unseen",
        "selection": "all val_unseen episodes in source order",
        "episode_ids": [str(ep.episode_id) for ep in dataset.episodes],
        "scene_ids": [str(ep.scene_id) for ep in dataset.episodes],
    }
    assert len(set(manifest["episode_ids"])) == len(dataset.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(dataset.episodes)} episodes, {len(set(manifest['scene_ids']))} scenes")


if __name__ == "__main__":
    main()
