"""Evaluate an already-frozen R2R-train scene screen with ActiveVLN.

Derived from tools/eval_val_seen_subset.py in the authorized ActiveVLN
checkout, SHA256 fec2cd3c0ff8912319dacca5b5616d5b6b850d96cced926980a0bc33037607b5.
Run from the checkout root with PYTHONPATH including vlnce_server. This
evaluator only reads a frozen manifest; it never selects or rewrites episodes.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--role", choices=["development", "reserved"], required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004/v1")
    args = parser.parse_args()
    if args.count != 256 or args.shard_count <= 0 or \
            not 0 <= args.shard_index < args.shard_count:
        raise ValueError("frozen screen requires 256 episodes and a valid shard")

    config_path = "vlnce_server/VLN_CE/vlnce_baselines/config/r2r_baselines/activevln_r2r_test.yaml"
    config = get_config(config_path)
    assert config.TASK_CONFIG.DATASET.SPLIT == "val_unseen"
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = "train"
    config.EVAL.SPLIT = "train"
    config.freeze()
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "trajectory_sil_rl_train_screen_v1" or \
            manifest["split"] != "train" or manifest["role"] != args.role or \
            manifest["source_sha256"] != "340a80133b2157520354ab055a91d98feb2f42e4bbda17b200c911f8788492ea":
        raise ValueError("screen schema, split, role, or source hash differs")
    ids = [str(episode_id) for episode_id in manifest["episode_ids"]]
    if len(ids) != args.count or len(set(ids)) != args.count:
        raise ValueError("screen episode coverage differs")
    selected_all = [by_id[episode_id] for episode_id in ids]
    scene_ids = [str(ep.scene_id).replace("data/scene_datasets/", "", 1)
                 for ep in selected_all]
    if scene_ids != manifest["scene_ids"] or len(set(scene_ids)) != 8:
        raise ValueError("screen scene coverage differs")
    chosen = selected_all[args.shard_index::args.shard_count]
    dataset.episodes = chosen

    if args.validate_only:
        print(json.dumps({"role": args.role, "split": "train", "manifest_count": len(ids),
                          "shard_count": len(chosen), "scene_count": len(set(scene_ids))}))
        return

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
        "split": "train",
        "role": args.role,
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
