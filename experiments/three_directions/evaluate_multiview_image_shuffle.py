"""Fixed same-scene pair-image permutation for selected two-view scorer."""

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
from train_multiview_event_lora import (CATEGORIES, evaluate, load_part,
                                        read, write)


SALT = "multiview-event-pair-shuffle-v1:"


def shuffled(items: list[dict]) -> tuple[list[dict], str, int]:
    clones = []
    for item in items:
        clone = dict(item)
        clone["record"] = copy.deepcopy(item["record"])
        clones.append(clone)
    by_scene = defaultdict(list)
    for index, item in enumerate(items):
        by_scene[item["plan"]["scene_id"]].append(index)
    mapping = []
    same_episode = 0
    for scene, indices in sorted(by_scene.items()):
        indices.sort(key=lambda i: hashlib.sha256((SALT + items[i][
            "label"]["pair_id"]).encode()).hexdigest())
        if len(indices) < 4:
            raise ValueError(f"too few development pairs in scene: {scene}")
        rotated = indices[len(indices)//2:] + indices[:len(indices)//2]
        for target_index, source_index in zip(indices, rotated):
            target = clones[target_index]
            source = items[source_index]
            if target["label"]["pair_id"] == source["label"]["pair_id"]:
                raise ValueError("identity shuffled pair")
            same_episode += (str(target["plan"]["episode_id"]) ==
                             str(source["plan"]["episode_id"]))
            for position in ("before", "after"):
                target_state = str(target["label"][f"{position}_state_index"])
                source_state = str(source["label"][f"{position}_state_index"])
                target["record"]["input"]["images_by_state"][target_state] = \
                    source["record"]["input"]["images_by_state"][source_state]
            mapping.append((target["label"]["pair_id"],
                            source["label"]["pair_id"]))
    if len(mapping) != len(items):
        raise ValueError("incomplete same-scene pair permutation")
    mapping_sha = hashlib.sha256(json.dumps(
        mapping, separators=(",", ":")).encode()).hexdigest()
    return clones, mapping_sha, same_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("capture-manifest", "pair-labels", "verification",
                 "rgb-root", "model", "checkpoint", "development",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    capture, labels, verified, report = map(
        read, (args.capture_manifest, args.pair_labels, args.verification,
               args.development))
    capture_sha = digest(args.capture_manifest)
    if labels["capture_manifest_sha256"] != capture_sha or \
            verified["capture_manifest_sha256"] != capture_sha or \
            report["capture_manifest_sha256"] != capture_sha or \
            report["audit_opened"] or report["navigation_result"]:
        raise ValueError("changed or opened source")
    selected = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False)
    if selected["capture_manifest_sha256"] != capture_sha or \
            selected["selected_step"] != report["selected_step"]:
        raise ValueError("selected two-view checkpoint changed")
    part = load_part("development", capture, labels, args.rgb_root,
                     capture_sha)
    items = [x for kind in CATEGORIES for x in part[kind]]
    randomized, mapping_sha, same_episode = shuffled(items)
    torch.set_num_threads(6)
    processor, model, head, _ = load_model(args.model)
    set_peft_model_state_dict(model, selected["adapter"])
    head.load_state_dict(selected["head"])
    metrics, scored = evaluate(processor, model, head, randomized, full=True)
    threshold = report["development"]["threshold"]["threshold"]
    pos = [x for x in scored if x["class"] == "crossing"]
    neg = [x for x in scored if x["class"] != "crossing"]
    wrong = [x for x in scored if x["class"] == "wrong_instruction"]
    retreat = [x for x in scored if x["class"] == "retreat"]
    near = [x for x in pos if not x["task_success_for_audit_only"]]
    value = {
        "schema": "multiview_event_image_shuffle_v1",
        "capture_manifest_sha256": capture_sha,
        "selected_step": selected["selected_step"],
        "checkpoint_sha256": digest(args.checkpoint),
        "permutation_salt": SALT,
        "permutation_sha256": mapping_sha,
        "same_scene_pair_permutation": True,
        "same_episode_assignments": same_episode,
        "pairs": len(items), "scenes": len({x["scene_id"] for x in scored}),
        "original_threshold": threshold,
        "auc_pooled": metrics["auc_pooled"],
        "within_route_rank_accuracy": metrics[
            "within_route_rank_accuracy"],
        "fixed_threshold_crossing_recall": sum(x["score"] >= threshold
            for x in pos)/len(pos),
        "fixed_threshold_pooled_fpr": sum(x["score"] >= threshold for x
            in neg)/len(neg),
        "fixed_threshold_retreat_fpr": sum(x["score"] >= threshold for x
            in retreat)/len(retreat),
        "fixed_threshold_wrong_fpr": sum(x["score"] >= threshold for x
            in wrong)/len(wrong),
        "fixed_threshold_near_failure_recall": sum(x["score"] >= threshold
            for x in near)/len(near),
        "audit_opened": False, "navigation_result": False,
    }
    write(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
