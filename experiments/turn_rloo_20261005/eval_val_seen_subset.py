"""Evaluate fixed, scene-balanced R2R val-seen episodes with ActiveVLN's evaluator.

Run from the root of an ActiveVLN checkout with PYTHONPATH including vlnce_server.
The first run writes a manifest. Later arms load exactly the same episode IDs.
"""

import argparse
import json
import os
import queue
import sys
import types
from pathlib import Path
from urllib.request import Request, urlopen

import habitat


# The Habitat simulator environment lacks the OpenAI and qwen-vl-utils Python
# packages used by ActiveVLN's evaluator. Supply the two small interfaces it
# calls without changing that environment's pinned Habitat dependencies.
if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, api_key, base_url):
            self.base_url = base_url.rstrip("/")
            self.models = types.SimpleNamespace(list=self.list_models)
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.complete)
            )

        def _request(self, path, body=None):
            request = Request(
                self.base_url + path,
                data=None if body is None else json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=300) as response:
                return json.load(response)

        def list_models(self):
            result = self._request("/models")
            return types.SimpleNamespace(data=[types.SimpleNamespace(id=x["id"]) for x in result["data"]])

        def complete(self, **kwargs):
            result = self._request("/chat/completions", kwargs)
            choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=x["message"]["content"]))
                       for x in result["choices"]]
            return types.SimpleNamespace(choices=choices)

    openai_module.OpenAI = OpenAI
    sys.modules["openai"] = openai_module

vision_module = types.ModuleType("qwen_vl_utils.vision_process")


def smart_resize(height, width, max_pixels, factor):
    scale = min(1.0, (max_pixels / (height * width)) ** 0.5)
    new_height = max(factor, round(height * scale / factor) * factor)
    new_width = max(factor, round(width * scale / factor) * factor)
    while new_height * new_width > max_pixels:
        if new_height >= new_width:
            new_height -= factor
        else:
            new_width -= factor
    return new_height, new_width


vision_module.smart_resize = smart_resize
qwen_module = types.ModuleType("qwen_vl_utils")
qwen_module.vision_process = vision_module
sys.modules["qwen_vl_utils"] = qwen_module
sys.modules["qwen_vl_utils.vision_process"] = vision_module

sys.path.insert(0, str(Path.cwd() / "eval" / "vlnce"))
from eval_vlnce import evaluate_agent  # noqa: E402
from VLN_CE.vlnce_baselines.config.default import get_config  # noqa: E402


def choose_scene_balanced(episodes, count):
    groups = {}
    for episode in episodes:
        groups.setdefault(str(episode.scene_id), []).append(episode)
    selected = []
    for round_index in range(max(map(len, groups.values()))):
        for scene in sorted(groups):
            if round_index < len(groups[scene]):
                selected.append(groups[scene][round_index])
                if len(selected) == count:
                    return selected
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004/v1")
    args = parser.parse_args()
    assert args.shard_count > 0 and 0 <= args.shard_index < args.shard_count

    config_path = "vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml"
    config = get_config(config_path)
    assert config.TASK_CONFIG.DATASET.SPLIT == "val_unseen"
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = "val_seen"
    config.EVAL.SPLIT = "val_seen"
    config.freeze()
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["split"] == "val_seen"
        selected_all = [by_id[episode_id] for episode_id in manifest["episode_ids"]]
        assert len(selected_all) == (len(dataset.episodes) if args.all_episodes else args.count)
    else:
        selected_all = (dataset.episodes if args.all_episodes
                        else choose_scene_balanced(dataset.episodes, args.count))
        assert len(selected_all) == (len(dataset.episodes) if args.all_episodes else args.count)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "split": "val_seen",
            "selection": ("all val_seen episodes in source order" if args.all_episodes
                          else "one per scene in sorted scene order, then second per scene"),
            "episode_ids": [str(ep.episode_id) for ep in selected_all],
            "scene_ids": [str(ep.scene_id) for ep in selected_all],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    chosen = selected_all[args.shard_index::args.shard_count]
    dataset.episodes = chosen

    result_path = Path(args.result_root) / args.model_label
    if args.shard_count > 1:
        result_path = result_path / f"shard_{args.shard_index:02d}"
    result_path.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = args.base_url
    q = queue.Queue()
    for retry in range(3):
        evaluate_agent(
            q, "r2r", "EMPTY", args.base_url, config, dataset,
            str(result_path), 1, 76800, args.max_turns,
        )
        rows = []
        invalid = []
        for episode in chosen:
            path = result_path / "log" / f"stats_{episode.episode_id}_0.json"
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                invalid.append(path)
                continue
            if row.get("early_stop_reason") == "inference_error":
                invalid.append(path)
            else:
                rows.append(row)
        if not invalid:
            break
        if retry == 2:
            raise RuntimeError(f"{len(invalid)} inference or result-file errors remain in shard {args.shard_index}")
        for path in invalid:
            if path.exists():
                archived = path.with_name(f"{path.name}.failed_retry{retry + 1}_pid{os.getpid()}")
                os.replace(path, archived)
        print(f"retrying {len(invalid)} incomplete or inference-error episodes in shard {args.shard_index}", flush=True)
    assert len(rows) == len(chosen)
    summary = {
        "label": args.model_label,
        "split": "val_seen",
        "count": len(rows),
        "successes": sum(bool(r["success"]) for r in rows),
        "sr": sum(bool(r["success"]) for r in rows) / len(rows),
        "spl": sum(float(r["spl"]) for r in rows) / len(rows),
        "oracle_sr": sum(bool(r["oracle_success"]) for r in rows) / len(rows),
        "mean_distance_to_goal": sum(float(r["distance_to_goal"]) for r in rows) / len(rows),
        "inference_errors": sum(r.get("early_stop_reason") == "inference_error" for r in rows),
        "max_turns": args.max_turns,
        "episode_ids": [str(ep.episode_id) for ep in chosen],
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    (result_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
