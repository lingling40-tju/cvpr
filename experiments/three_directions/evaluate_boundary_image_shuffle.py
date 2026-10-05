"""Frozen same-scene RGB permutation control for the selected occupancy head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

from peft import set_peft_model_state_dict
import torch

from history_grounding_lora import digest, load_model
from train_boundary_occupancy_lora import evaluate, load_part, read, write


SALT = "boundary-occupancy-image-shuffle-v1:"


def shuffled(items: list[dict]) -> tuple[list[dict], str]:
    clones = []
    for item in items:
        clone = dict(item)
        clone["record"] = copy.deepcopy(item["record"])
        clones.append(clone)
    by_scene = defaultdict(list)
    for index, item in enumerate(items):
        for state in ("outside", "inside"):
            by_scene[item["plan"]["scene_id"]].append((index, state))
    mappings = []
    for scene, slots in sorted(by_scene.items()):
        slots.sort(key=lambda slot: hashlib.sha256(
            (SALT + items[slot[0]]["plan"]["record_id"] + ":" +
             slot[1]).encode()).hexdigest())
        if len(slots) < 4 or len(slots) % 2:
            raise ValueError(f"insufficient paired scene slots: {scene}")
        shifted = slots[len(slots)//2:] + slots[:len(slots)//2]
        for (target_index, target_state), (source_index, source_state) in \
                zip(slots, shifted):
            target = clones[target_index]
            source = items[source_index]
            path = source["record"]["input"]["images"][source_state]
            target["record"]["input"]["images"][target_state] = path
            mappings.append((target["plan"]["record_id"], target_state,
                             source["plan"]["record_id"], source_state))
    if len(mappings) != len(items)*2 or any(a == c and b == d
                                              for a, b, c, d in mappings):
        raise ValueError("incomplete or identity RGB permutation")
    return clones, hashlib.sha256(json.dumps(mappings,
        separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "verification", "replay-root",
                 "model", "checkpoint", "development", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest, labels, verification, development = map(
        read, (args.manifest, args.labels, args.verification,
               args.development))
    manifest_sha = digest(args.manifest)
    if labels["replay_manifest_sha256"] != manifest_sha or \
            verification["manifest_sha256"] != manifest_sha or \
            development["manifest_sha256"] != manifest_sha or \
            development["audit_opened"] or \
            development["navigation_result"]:
        raise ValueError("changed source or development selection")
    selected = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False)
    if selected["manifest_sha256"] != manifest_sha or \
            selected["selected_step"] != development["selected_step"]:
        raise ValueError("selected occupancy checkpoint changed")
    items = load_part("development", manifest, labels, args.replay_root,
                      manifest_sha)
    shuffled_items, mapping_sha = shuffled(items)
    torch.set_num_threads(6)
    processor, model, head, _ = load_model(args.model)
    set_peft_model_state_dict(model, selected["adapter"])
    head.load_state_dict(selected["head"])
    fitted, scores = evaluate(processor, model, head, shuffled_items,
                              full=True)
    threshold = development["development"]["threshold"]["threshold"]
    positive = [r for r in scores if r["class"] == "inside"]
    negative = [r for r in scores if r["class"] != "inside"]
    wrong = [r for r in scores if r["class"] == "wrong"]
    near_failure = [r for r in positive if not r[
        "task_success_for_audit_only"]]
    report = {"schema": "boundary_occupancy_image_shuffle_v1",
              "manifest_sha256": manifest_sha,
              "selected_step": selected["selected_step"],
              "checkpoint_sha256": digest(args.checkpoint),
              "permutation_salt": SALT,
              "permutation_sha256": mapping_sha,
              "scenes": len({r["scene_id"] for r in scores}),
              "records": len(items), "image_slots_permuted": len(items)*2,
              "same_scene_permutation": True,
              "original_threshold": threshold,
              "auc_pooled": fitted["auc_pooled"],
              "crossing_order_accuracy": fitted["rates"][
                  "crossing_order_accuracy"],
              "instruction_order_accuracy": fitted["rates"][
                  "instruction_order_accuracy"],
              "fixed_threshold_pooled_recall": sum(
                  r["score"] >= threshold for r in positive)/len(positive),
              "fixed_threshold_pooled_fpr": sum(
                  r["score"] >= threshold for r in negative)/len(negative),
              "fixed_threshold_wrong_fpr": sum(
                  r["score"] >= threshold for r in wrong)/len(wrong),
              "fixed_threshold_near_failure_recall": sum(
                  r["score"] >= threshold for r in near_failure)/len(
                      near_failure),
              "audit_opened": False, "navigation_result": False}
    write(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
