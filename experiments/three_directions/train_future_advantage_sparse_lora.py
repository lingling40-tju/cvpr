"""Fit an observation-only within-group future-advantage ranker.

The only tensors sent to the model come from sparse RGB, instruction and
executed actions. Privileged future returns select pair labels outside the
model call. The fixed final checkpoint is evaluated once on train-scene
development pairs; it is not chosen using that development result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import time

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from future_advantage_visual_input import build_inputs


ANCHORS = (3, 6)
MICROSTEPS = 1024
ACCUMULATION = 4
SEED = 11
LORA = dict(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=["q_proj", "v_proj"])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(manifest_path: Path, labels_path: Path,
              replay_root: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(labels_path.read_text())
    if manifest.get("schema") != "future_advantage_sparse_replay_manifest_v1" or \
            labels.get("schema") != "future_advantage_within_group_pair_labels_v1" or \
            manifest.get("group_size") != 4 or labels.get("group_size") != 4 or \
            manifest.get("anchors") != [3, 6] or labels.get("anchors") != [3, 6] or \
            manifest.get("seeds") != labels.get("seeds") or \
            labels.get("replay_manifest_sha256") != digest(manifest_path) or \
            labels.get("preflight_report_sha256") != manifest.get(
                "preflight_report_sha256") or \
            manifest.get("counts") != labels.get("counts"):
        raise ValueError("mismatched or non-n=4 sparse replay sources")
    records, pairs, metadata = {}, {}, {}
    scenes = {}
    for part in ("fit", "development"):
        records[part] = {}
        metadata[part] = {}
        scenes[part] = set()
        for selected in manifest["selected"][part]:
            rid = selected["record_id"]
            if rid in records[part] or selected["seed"] not in manifest["seeds"]:
                raise ValueError("duplicate record or unselected seed")
            path = replay_root / part / "records" / f"{rid}.json"
            record = json.loads(path.read_text())
            if record.get("record_id") != rid or record.get(
                    "manifest_sha256") != digest(manifest_path) or \
                    record.get("schema") != "future_advantage_sparse_model_input_v1":
                raise ValueError(f"invalid sparse replay: {rid}")
            records[part][rid] = record
            metadata[part][rid] = selected
            scenes[part].add(selected["scene_id"])
        pairs[part] = []
        seen = set()
        for row in labels["pairs"][part]:
            if row["pair_id"] in seen:
                raise ValueError("duplicate pair label")
            seen.add(row["pair_id"])
            anchor, seed, eid = row["anchor"], row["seed"], str(row["episode_id"])
            if anchor not in ANCHORS or seed not in manifest["seeds"] or \
                    row["preferred_variant"] not in (
                        row["left_variant"], row["right_variant"]):
                raise ValueError("invalid pair identity")
            left = f"s{seed}_e{eid}_v{row['left_variant']}"
            right = f"s{seed}_e{eid}_v{row['right_variant']}"
            if left not in records[part] or right not in records[part]:
                raise ValueError("pair has missing sparse RGB record")
            for rid, variant in ((left, row["left_variant"]),
                                 (right, row["right_variant"])):
                selected = metadata[part][rid]
                if selected["seed"] != seed or \
                        str(selected["episode_id"]) != eid or \
                        selected["scene_id"] != row["scene_id"] or \
                        selected["variant"] != variant:
                    raise ValueError("pair and replay identity disagree")
            if bool(row["same_terminal_mode"]) != (
                    metadata[part][left]["terminal_mode"] ==
                    metadata[part][right]["terminal_mode"]):
                raise ValueError("pair terminal-mode label disagrees with replay")
            gap = float(row["future_return_gap_for_label_only"])
            if abs(gap) < .25 or (gap > 0) != (
                    row["preferred_variant"] == row["left_variant"]):
                raise ValueError("preferred route contradicts privileged label")
            pairs[part].append({
                "left": left, "right": right, "sign": 1.0 if gap > 0 else -1.0,
                "anchor": anchor, "scene": row["scene_id"], "episode": eid,
                "seed": seed, "same_mode": bool(row["same_terminal_mode"]),
            })
        for anchor in ANCHORS:
            section = manifest["counts"][part][str(anchor)]
            rows = [row for row in pairs[part] if row["anchor"] == anchor]
            same = [row for row in rows if row["same_mode"]]
            if (len(rows), len({r["episode"] for r in rows}), len(same),
                    len({r["episode"] for r in same})) != (
                    section["pairs"], section["unique_episode_groups"],
                    section["same_terminal_mode_pairs"],
                    section["same_terminal_mode_unique_episode_groups"]):
                raise ValueError(f"pair count mismatch: {part}/{anchor}")
    if scenes["fit"] & scenes["development"]:
        raise ValueError("fit and development scenes overlap")
    return records, pairs, {"manifest": digest(manifest_path),
                            "labels": digest(labels_path)}


class ScalarHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, 128), nn.GELU(),
                                 nn.Linear(128, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.float()).squeeze(-1)


def load_model(path: Path):
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True,
                                              use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, local_files_only=True)
    base.config.use_cache = False
    base.lm_head = nn.Identity()
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, LoraConfig(**LORA)).cuda()
    model.enable_input_require_grads()
    head = ScalarHead(base.config.hidden_size).cuda()
    return processor, model, head


def score(processor, model, head, record: dict, root: Path,
          anchor: int) -> torch.Tensor:
    inputs = build_inputs(processor, record, root, anchor).to("cuda")
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    return head(output.logits[0, -1])


def forward_meters(record: dict, anchor: int) -> float:
    history = record["input"]["action_history_by_anchor"][str(anchor)]
    meters = 0.0
    for turn in history:
        for action in turn["executed_actions"]:
            if action.startswith("move forward "):
                match = re.fullmatch(r"move forward (\d+)cm", action)
                if match is None:
                    raise ValueError(f"unrecognized forward action: {action}")
                meters += int(match.group(1)) / 100.0
    return meters


def macros(rows: list[tuple[dict, float, float]]) -> dict:
    by_episode = defaultdict(list)
    baseline_episode = defaultdict(list)
    for row, prediction, baseline in rows:
        key = (row["scene"], row["episode"])
        by_episode[key].append(prediction)
        baseline_episode[key].append(baseline)
    by_scene = defaultdict(list)
    for (scene, _), values in by_episode.items():
        by_scene[scene].append(sum(values) / len(values))
    scene_means = [sum(values) / len(values) for _, values in
                   sorted(by_scene.items())]
    interval = None
    if len(scene_means) >= 2:
        rng = random.Random(SEED)
        draws = sorted(sum(rng.choices(scene_means, k=len(scene_means))) /
                       len(scene_means) for _ in range(5000))
        interval = [draws[124], draws[4874]]
    return {
        "pairs": len(rows), "unique_episode_ids": len(by_episode),
        "scenes": len(by_scene),
        "episode_macro": sum(sum(v) / len(v) for v in by_episode.values())
        / len(by_episode) if by_episode else None,
        "scene_macro": sum(sum(v) / len(v) for v in by_scene.values())
        / len(by_scene) if by_scene else None,
        "scene_bootstrap95": interval,
        "forward_action_episode_macro": sum(
            sum(v) / len(v) for v in baseline_episode.values())
        / len(baseline_episode) if baseline_episode else None,
    }


def evaluate(processor, model, head, records: dict, pairs: list[dict],
             root: Path) -> dict:
    model.eval()
    head.eval()
    cache = {}
    result = {}
    with torch.inference_mode():
        for anchor in ANCHORS:
            scored = []
            for row in pairs:
                if row["anchor"] != anchor:
                    continue
                values = []
                forward = []
                for rid in (row["left"], row["right"]):
                    key = (rid, anchor)
                    if key not in cache:
                        cache[key] = float(score(
                            processor, model, head, records[rid], root, anchor))
                    values.append(cache[key])
                    forward.append(forward_meters(records[rid], anchor))
                signed = row["sign"] * (values[0] - values[1])
                # Extra forward motion is a deliberately weak action-only baseline.
                baseline = row["sign"] * (forward[0] - forward[1])
                scored.append((row, 1.0 if signed > 0 else (.5 if signed == 0 else 0.0),
                               1.0 if baseline > 0 else (.5 if baseline == 0 else 0.0)))
            all_score = macros(scored)
            same_score = macros([x for x in scored if x[0]["same_mode"]])
            different_score = macros([x for x in scored if not x[0]["same_mode"]])
            result[str(anchor)] = {"all": all_score,
                                   "same_terminal_mode": same_score,
                                   "different_terminal_mode": different_score,
                                   "by_forward_action_baseline": {
                                       name: macros([x for x in scored if
                                                     x[2] == value])
                                       for name, value in (("correct", 1.0),
                                                           ("tie", .5),
                                                           ("incorrect", 0.0))},
                                   "by_preferred_side": {
                                       side: macros([x for x in scored
                                                     if x[0]["sign"] == sign])
                                       for side, sign in (("left", 1), ("right", -1))}}
    checks = {}
    for anchor in ANCHORS:
        block = result[str(anchor)]
        all_score, same_score = block["all"], block["same_terminal_mode"]
        checks[str(anchor)] = bool(
            all_score["episode_macro"] is not None and
            same_score["episode_macro"] is not None and
            all_score["episode_macro"] >= .70 and
            same_score["episode_macro"] >= .70 and
            all_score["scene_macro"] >= .65 and
            same_score["scene_macro"] >= .65 and
            all_score["episode_macro"] -
            all_score["forward_action_episode_macro"] >= .05 and
            same_score["episode_macro"] -
            same_score["forward_action_episode_macro"] >= .05)
    return {"anchors": result, "development_gate": checks,
            "passed": all(checks.values()), "unique_scored_prefixes": len(cache)}


def synthetic_records(root: Path) -> tuple[dict, list[dict]]:
    records = {}
    for variant in (0, 1):
        images = {}
        for anchor in (0, 3, 6):
            path = root / f"v{variant}_a{anchor}.png"
            Image.new("RGB", (160, 120),
                      (40 + variant * 110, 50 + anchor * 20, 70)).save(path)
            images[str(anchor)] = path.name
        history = [{"turn": turn, "executed_actions": [
            "move forward 25cm" if variant == 0 else "turn left 15deg"]}
            for turn in range(1, 7)]
        records[str(variant)] = {
            "schema": "future_advantage_sparse_model_input_v1",
            "input": {"instruction": "Walk toward the red chair.",
                      "images": images,
                      "action_history_by_anchor": {
                          "3": history[:3], "6": history}}}
    return records, [{"left": "0", "right": "1", "sign": 1.0,
                      "anchor": anchor, "episode": "synthetic",
                      "scene": "synthetic", "seed": 11, "same_mode": True}
                     for anchor in ANCHORS]


def train(args) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(6)
    rng = random.Random(SEED)
    if args.synthetic_smoke:
        temporary = tempfile.TemporaryDirectory(prefix="fa_lora_smoke_")
        root = Path(temporary.name)
        fit_records, fit_pairs = synthetic_records(root)
        source = {"synthetic": True}
        steps = 6
    else:
        if args.manifest is None or args.labels is None or args.replay_root is None:
            raise ValueError("real fit requires gated manifest, labels and replay root")
        records, pairs, source = load_data(
            args.manifest, args.labels, args.replay_root)
        fit_records, fit_pairs = records["fit"], pairs["fit"]
        root = args.replay_root / "fit"
        steps = MICROSTEPS
    fit_index = {anchor: defaultdict(lambda: defaultdict(list))
                 for anchor in ANCHORS}
    for row in fit_pairs:
        if row["same_mode"]:
            fit_index[row["anchor"]][row["episode"]][row["seed"]].append(row)
    if any(not fit_index[anchor] for anchor in ANCHORS):
        raise ValueError("missing same-terminal-mode fit pairs at an anchor")
    processor, model, head = load_model(args.model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": params, "lr": 5e-5},
        {"params": head.parameters(), "lr": 1.5e-4}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(1, steps + 1):
        anchor = ANCHORS[(step - 1) % 2]
        episode = rng.choice(sorted(fit_index[anchor]))
        by_seed = fit_index[anchor][episode]
        seed = rng.choice(sorted(by_seed))
        row = rng.choice(by_seed[seed])
        model.train()
        head.train()
        left = score(processor, model, head, fit_records[row["left"]],
                     root, anchor)
        right = score(processor, model, head, fit_records[row["right"]],
                      root, anchor)
        loss = F.softplus(-row["sign"] * (left - right)) + \
            .001 * (left.square() + right.square())
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite ranking loss at step {step}")
        (loss / ACCUMULATION).backward()
        if step % ACCUMULATION == 0:
            norm = torch.nn.utils.clip_grad_norm_(params + list(head.parameters()),
                                                   1.0)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0:
                raise ValueError(f"invalid gradient at step {step}: {norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 16 == 0 or step == steps:
            print(json.dumps({"step": step, "anchor": anchor,
                              "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}), flush=True)
    if args.synthetic_smoke:
        if not any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in params):
            raise ValueError("synthetic final microsteps have no finite LoRA gradients")
        print(json.dumps({"schema": "future_advantage_sparse_synthetic_smoke_v1",
                          "microsteps": steps, "optimizer_updates": 1,
                          "real_rgb_or_navigation_result": False,
                          "elapsed_seconds": time.time() - started}), flush=True)
        temporary.cleanup()
        return
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "future_advantage_sparse_lora_final_v1",
                "microsteps": MICROSTEPS, "accumulation": ACCUMULATION,
                "seed": SEED, "sources_sha256": source,
                "model_config_sha256": digest(args.model / "config.json"),
                "trainer_sha256": digest(Path(__file__)),
                "visual_input_sha256": digest(Path(__file__).with_name(
                    "future_advantage_visual_input.py")),
                "adapter": {k: v.detach().cpu() for k, v in
                            get_peft_model_state_dict(model).items()},
                "head": {k: v.detach().cpu() for k, v in
                         head.state_dict().items()}}, args.output / "adapter_head.pt")
    metrics = evaluate(processor, model, head, records["development"],
                       pairs["development"], args.replay_root / "development")
    result = {"schema": "future_advantage_sparse_development_v1",
              "source_sha256": source, "fit_microsteps": MICROSTEPS,
              "fit_pair_count": len([r for r in fit_pairs if r["same_mode"]]),
              "development": metrics,
              "elapsed_seconds": time.time() - started,
              "interpretation": "Train-scene development gate, not a val-unseen navigation result"}
    (args.output / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke and args.output is None:
        parser.error("--output is required for a real fit")
    train(args)


if __name__ == "__main__":
    main()
