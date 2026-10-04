"""Fit a current-distance potential from observation-only group-four prefixes.

Simulator distance appears only in pair construction and offline metrics.
The model input contains instruction, available RGB, and executed actions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F
from peft import get_peft_model_state_dict

from anchor_potential_visual_input import build_inputs
from train_future_advantage_sparse_lora import (
    ACCUMULATION, ANCHORS, MICROSTEPS, SEED, digest, forward_meters,
    load_model, macros, synthetic_records,
)
from verify_future_advantage_sparse_replay import verify as verify_sparse_replay


def load_anchor_pairs(manifest_path: Path, replay_root: Path,
                      preflight_path: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    manifest_sha = digest(manifest_path)
    if manifest.get("schema") != "future_advantage_sparse_replay_manifest_v1" or \
            manifest.get("group_size") != 4 or manifest.get("seeds") != [11, 22] or \
            preflight.get("schema") != "dense_instruction_potential_train_source_preflight_v1" or \
            preflight.get("source_manifest_sha256") != manifest_sha or \
            preflight.get("train_dataset_sha256") != manifest["source_sha256"]["dataset"]:
        raise ValueError("changed or non-group-four anchor source")
    records, pairs, scenes = {}, {}, {}
    for part in ("fit", "development"):
        records[part] = {}
        scenes[part] = set()
        grouped = defaultdict(list)
        for plan in manifest["selected"][part]:
            rid = plan["record_id"]
            record = json.loads((replay_root / part / "records" /
                                 f"{rid}.json").read_text())
            audit = json.loads((replay_root / part / "audits" /
                                f"{rid}.json").read_text())
            if rid in records[part] or record.get("record_id") != rid or \
                    audit.get("record_id") != rid or \
                    record.get("manifest_sha256") != manifest_sha or \
                    audit.get("manifest_sha256") != manifest_sha or \
                    audit.get("seed") != plan["seed"] or \
                    str(audit.get("episode_id")) != str(plan["episode_id"]) or \
                    audit.get("variant") != plan["variant"]:
                raise ValueError(f"replay identity mismatch: {rid}")
            records[part][rid] = record
            scenes[part].add(plan["scene_id"])
            grouped[(plan["seed"], str(plan["episode_id"]))].append(
                (plan, audit))
        pairs[part] = []
        for (seed, eid), group in sorted(grouped.items()):
            group.sort(key=lambda row: row[0]["variant"])
            for index, (left_plan, left_audit) in enumerate(group):
                for right_plan, right_audit in group[index + 1:]:
                    if left_plan["scene_id"] != right_plan["scene_id"]:
                        raise ValueError("cross-scene same-episode pair")
                    for anchor in ANCHORS:
                        if anchor not in left_plan["anchor_turns"] or \
                                anchor not in right_plan["anchor_turns"]:
                            continue
                        left_turn = next((turn for turn in left_audit["turns"]
                                          if turn["turn"] == anchor), None)
                        right_turn = next((turn for turn in right_audit["turns"]
                                           if turn["turn"] == anchor), None)
                        if left_turn is None or right_turn is None:
                            raise ValueError("missing audited anchor distance")
                        left_distance = float(left_turn["after_distance_m"])
                        right_distance = float(right_turn["after_distance_m"])
                        gap = right_distance - left_distance
                        if abs(gap) < 1.0:
                            continue
                        pairs[part].append({
                            "left": left_plan["record_id"],
                            "right": right_plan["record_id"],
                            "sign": 1.0 if gap > 0 else -1.0,
                            "anchor": anchor, "seed": seed,
                            "episode": eid, "scene": left_plan["scene_id"],
                            "same_mode": left_plan["terminal_mode"] ==
                            right_plan["terminal_mode"],
                        })
        for anchor in ANCHORS:
            rows = [row for row in pairs[part] if row["anchor"] == anchor]
            same = [row for row in rows if row["same_mode"]]
            expected = preflight["parts"][part]["same_episode_anchor_pairs"][
                str(anchor)]
            if len(rows) != expected["distance_gap_ge_1m"] or \
                    len(same) != expected["same_terminal_mode"] or \
                    len({(r["seed"], r["episode"]) for r in rows}) != \
                    expected["unique_seed_episode_groups"]:
                raise ValueError(f"anchor pair coverage changed: {part}/{anchor}")
    if scenes["fit"] & scenes["development"]:
        raise ValueError("fit/development scenes overlap")
    return records, pairs, {"manifest": manifest_sha,
                            "preflight": digest(preflight_path)}


def score(processor, model, head, record: dict, root: Path,
          anchor: int) -> torch.Tensor:
    inputs = build_inputs(processor, record, root, anchor).to("cuda")
    output = model(**inputs, output_hidden_states=False, use_cache=False)
    return head(output.logits[0, -1])


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
                values, forward = [], []
                for rid in (row["left"], row["right"]):
                    key = (rid, anchor)
                    if key not in cache:
                        cache[key] = float(score(
                            processor, model, head, records[rid], root,
                            anchor))
                    values.append(cache[key])
                    forward.append(forward_meters(records[rid], anchor))
                signed = row["sign"] * (values[0] - values[1])
                baseline = row["sign"] * (forward[0] - forward[1])
                scored.append((
                    row, 1.0 if signed > 0 else (.5 if signed == 0 else 0.0),
                    1.0 if baseline > 0 else (.5 if baseline == 0 else 0.0)))
            all_score = macros(scored)
            same_score = macros([x for x in scored if x[0]["same_mode"]])
            result[str(anchor)] = {
                "all": all_score,
                "same_terminal_mode": same_score,
                "different_terminal_mode": macros(
                    [x for x in scored if not x[0]["same_mode"]]),
                "by_preferred_side": {
                    side: macros([x for x in scored if x[0]["sign"] == sign])
                    for side, sign in (("left", 1), ("right", -1))},
            }
    checks = {}
    for anchor in ANCHORS:
        all_score = result[str(anchor)]["all"]
        same_score = result[str(anchor)]["same_terminal_mode"]
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
            "passed": all(checks.values()),
            "unique_scored_prefixes": len(cache)}


def train(args) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(6)
    rng = random.Random(SEED)
    if args.synthetic_smoke:
        temporary = tempfile.TemporaryDirectory(prefix="anchor_potential_smoke_")
        root = Path(temporary.name)
        fit_records, fit_pairs = synthetic_records(root)
        source = {"synthetic": True}
        steps = 6
    else:
        if any(value is None for value in (args.manifest, args.labels,
                                          args.report, args.preflight,
                                          args.replay_root, args.output)):
            raise ValueError("real fit requires frozen sources and output")
        verification = verify_sparse_replay(
            args.manifest, args.labels, args.report, args.replay_root)
        records, pairs, source = load_anchor_pairs(
            args.manifest, args.replay_root, args.preflight)
        if verification["source_sha256"]["manifest"] != source["manifest"]:
            raise ValueError("verified replay source changed")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "source_verification.json").write_text(
            json.dumps(verification, indent=2) + "\n")
        fit_records, fit_pairs = records["fit"], pairs["fit"]
        root = args.replay_root / "fit"
        steps = MICROSTEPS
    fit_index = {anchor: defaultdict(lambda: defaultdict(list))
                 for anchor in ANCHORS}
    for row in fit_pairs:
        if row["same_mode"]:
            fit_index[row["anchor"]][row["episode"]][row["seed"]].append(row)
    if any(not fit_index[anchor] for anchor in ANCHORS):
        raise ValueError("missing same-mode fit pair at an anchor")
    processor, model, head = load_model(args.model)
    params = [param for param in model.parameters() if param.requires_grad]
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
            raise ValueError(f"nonfinite pair loss at step {step}")
        (loss / ACCUMULATION).backward()
        if step % ACCUMULATION == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                params + list(head.parameters()), 1.0)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0:
                raise ValueError(f"invalid gradient at step {step}: {norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 16 == 0 or step == steps:
            print(json.dumps({"step": step, "anchor": anchor,
                              "loss": float(loss.detach()),
                              "elapsed_seconds": time.time() - started}),
                  flush=True)
    if args.synthetic_smoke:
        print(json.dumps({"schema": "anchor_potential_synthetic_smoke_v1",
                          "microsteps": steps,
                          "real_rgb_or_navigation_result": False,
                          "elapsed_seconds": time.time() - started}), flush=True)
        temporary.cleanup()
        return
    torch.save({
        "schema": "anchor_distance_potential_lora_final_v1",
        "microsteps": MICROSTEPS, "accumulation": ACCUMULATION,
        "seed": SEED, "sources_sha256": source,
        "model_config_sha256": digest(args.model / "config.json"),
        "trainer_sha256": digest(Path(__file__)),
        "visual_input_sha256": digest(Path(__file__).with_name(
            "anchor_potential_visual_input.py")),
        "adapter": {key: value.detach().cpu() for key, value in
                    get_peft_model_state_dict(model).items()},
        "head": {key: value.detach().cpu() for key, value in
                 head.state_dict().items()},
    }, args.output / "adapter_head.pt")
    metrics = evaluate(processor, model, head, records["development"],
                       pairs["development"], args.replay_root / "development")
    result = {
        "schema": "anchor_distance_potential_development_v1",
        "source_sha256": source, "fit_microsteps": MICROSTEPS,
        "fit_pair_count": sum(row["same_mode"] for row in fit_pairs),
        "development": metrics,
        "elapsed_seconds": time.time() - started,
        "interpretation": "Exploratory train-scene development; no navigation result",
    }
    (args.output / "development.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
