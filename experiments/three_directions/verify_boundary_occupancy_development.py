"""Independent standard-library recount of the frozen occupancy screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, reported: float) -> None:
    if not math.isfinite(actual) or abs(actual - reported) > 1e-9:
        raise ValueError(f"independent metric mismatch: {actual} != {reported}")


def macro(rows: list[dict], key: str, threshold: float) -> tuple[int, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row["score"] >= threshold)
    return len(groups), sum(sum(v)/len(v) for v in groups.values())/len(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "replay-verification", "report",
                 "scores", "history-only", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest, labels, replay, report, score_file, history = map(
        read, (args.manifest, args.labels, args.replay_verification,
               args.report, args.scores, args.history_only))
    manifest_sha = sha(args.manifest)
    if manifest["schema"] != "boundary_occupancy_rgb_replay_manifest_v1" or \
            labels["replay_manifest_sha256"] != manifest_sha or \
            replay["manifest_sha256"] != manifest_sha or \
            report["manifest_sha256"] != score_file["manifest_sha256"] != \
                history["manifest_sha256"] != manifest_sha or \
            score_file["selected_step"] != report["selected_step"] or \
            report["audit_opened"] or report["navigation_result"] or \
            history["image_files_opened"] != 0:
        raise ValueError("changed source or opened audit")
    plans = {p["record_id"]: p for p in manifest["selected"]["development"]}
    truth = {p["record_id"]: p for p in labels["selected"]["development"]}
    if len(plans) != 288 or set(plans) != set(truth):
        raise ValueError("changed held development plan")
    by_id = defaultdict(dict)
    for row in score_file["scores"]:
        rid, cls = row["record_id"], row["class"]
        if rid not in plans or cls not in ("outside", "inside", "wrong") or \
                cls in by_id[rid] or \
                row["episode_id"] != str(plans[rid]["episode_id"]) or \
                row["scene_id"] != plans[rid]["scene_id"] or \
                row["task_success_for_audit_only"] != truth[rid][
                    "task_success"] or \
                row["same_start_wrong"] != plans[rid][
                    "wrong_instruction_same_start"] or \
                not math.isfinite(row["score"]):
            raise ValueError(f"changed development score row: {rid}/{cls}")
        by_id[rid][cls] = row
    if set(by_id) != set(plans) or any(
            set(v) != ({"outside", "inside", "wrong"} if
                       plans[rid]["wrong_instruction"] else
                       {"outside", "inside"}) for rid, v in by_id.items()):
        raise ValueError("missing or extra development score class")
    positive = [v["inside"] for v in by_id.values()]
    outside = [v["outside"] for v in by_id.values()]
    wrong = [v["wrong"] for v in by_id.values() if "wrong" in v]
    negative = outside + wrong
    candidates = sorted({r["score"] for r in positive + negative},
                        reverse=True)
    feasible = [(sum(r["score"] >= threshold for r in positive)/len(positive),
                 -sum(r["score"] >= threshold for r in negative)/len(negative),
                 threshold) for threshold in [max(candidates)+1] + candidates
                if sum(r["score"] >= threshold for r in negative)/
                   len(negative) <= .05]
    recall, neg_fpr, threshold = max(feasible)
    dev = report["development"]
    for actual, reported in ((recall, dev["threshold"]["pooled_recall"]),
                             (-neg_fpr, dev["threshold"]["pooled_fpr"]),
                             (threshold, dev["threshold"]["threshold"])):
        close(actual, reported)
    scores = {
        "pooled_recall": recall,
        "pooled_fpr": -neg_fpr,
        "wrong_instruction_fpr": sum(x["score"] >= threshold for x in
                                     wrong)/len(wrong),
        "near_failure_recall": sum(x["score"] >= threshold for x in
            positive if not x["task_success_for_audit_only"])/sum(
                not x["task_success_for_audit_only"] for x in positive),
        "crossing_order_accuracy": sum(v["inside"]["score"] >
            v["outside"]["score"] for v in by_id.values())/len(by_id),
        "instruction_order_accuracy": sum(v["inside"]["score"] >
            v["wrong"]["score"] for v in by_id.values() if "wrong" in v)/
                len(wrong),
    }
    for key in ("wrong_instruction_fpr", "near_failure_recall",
                "crossing_order_accuracy", "instruction_order_accuracy"):
        close(scores[key], dev["rates"][key])
    auc = sum((p["score"] > n["score"]) +
              .5 * (p["score"] == n["score"]) for p in positive
              for n in negative)/(len(positive)*len(negative))
    close(auc, dev["auc_pooled"])
    for scale, key in (("episode_macro", "episode_id"),
                       ("scene_macro", "scene_id")):
        for name, rows in (("inside_recall", positive),
                           ("outside_fpr", outside), ("wrong_fpr", wrong)):
            count, rate = macro(rows, key, threshold)
            entry = dev[scale][name]
            if count != entry["groups"]:
                raise ValueError("group count mismatch")
            close(rate, entry["macro_rate"])
    gates = {
        "positive_episode_ids_at_least_50":
            len({r["episode_id"] for r in positive}) >= 50,
        "pooled_fpr_at_most_0_05": -neg_fpr <= .05,
        "pooled_recall_at_least_0_55": recall >= .55,
        "wrong_instruction_fpr_at_most_0_12":
            scores["wrong_instruction_fpr"] <= .12,
        "near_failure_recall_at_least_0_50":
            scores["near_failure_recall"] >= .50,
    }
    if gates != dev["gate"] or report["gate_passed"] != all(gates.values()):
        raise ValueError("development gate mismatch")
    output = {
        "schema": "boundary_occupancy_independent_development_recount_v1",
        "manifest_sha256": manifest_sha,
        "development_records": len(by_id),
        "score_rows": len(score_file["scores"]),
        "selected_step": report["selected_step"],
        "auc_pooled": auc, "metrics": scores, "gates": gates,
        "gate_passed": all(gates.values()),
        "history_only_auc": history["auc_pooled"],
        "auc_gain_over_history_only": auc-history["auc_pooled"],
        "audit_opened": False, "navigation_result": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
