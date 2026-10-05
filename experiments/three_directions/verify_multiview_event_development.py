"""Independent, standard-library recount of the frozen two-view screen."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


KINDS = ("crossing", "far_nonarrival", "retreat", "wrong_instruction")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a: float, b: float) -> None:
    if not math.isfinite(a) or abs(a - b) > 1e-9:
        raise ValueError(f"metric mismatch: {a} != {b}")


def rate(rows: list[dict], threshold: float) -> float:
    return sum(x["score"] >= threshold for x in rows) / len(rows)


def auc(pos: list[dict], neg: list[dict]) -> float:
    return sum((p["score"] > n["score"]) +
               .5 * (p["score"] == n["score"])
               for p in pos for n in neg) / (len(pos) * len(neg))


def macro(rows: list[dict], key: str, threshold: float) -> tuple[int, float]:
    groups = defaultdict(list)
    for x in rows:
        groups[str(x[key])].append(x)
    return len(groups), sum(rate(v, threshold) for v in groups.values()) / len(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("capture-manifest", "pair-labels", "replay-verification",
                 "report", "scores", "text-only", "image-shuffle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    capture, labels, replay, report, scores, text_only, shuffled = map(
        read, (args.capture_manifest, args.pair_labels,
               args.replay_verification, args.report, args.scores,
               args.text_only, args.image_shuffle))
    capture_sha = sha(args.capture_manifest)
    if any(x["capture_manifest_sha256"] != capture_sha for x in
           (labels, replay, report, scores, text_only, shuffled)):
        raise ValueError("source digest changed")
    if report["audit_opened"] or report["navigation_result"] or \
            shuffled["audit_opened"] or shuffled["navigation_result"] or \
            text_only["image_files_opened"] != 0:
        raise ValueError("audit or navigation result opened")
    if not (report["selected_step"] == scores["selected_step"] ==
            shuffled["selected_step"]):
        raise ValueError("selected checkpoint mismatch")
    plans = {x["record_id"]: x for x in capture["selected"]["development"]}
    truth = {x["pair_id"]: x for x in labels["selected"]["development"]}
    if len(plans) != 351 or len(truth) != 758:
        raise ValueError("development source count changed")
    seen = set()
    by_kind = {kind: [] for kind in KINDS}
    by_record = defaultdict(dict)
    for row in scores["scores"]:
        pair_id = row["pair_id"]
        if pair_id in seen or pair_id not in truth:
            raise ValueError(f"extra or duplicate pair: {pair_id}")
        seen.add(pair_id)
        label = truth[pair_id]
        plan = plans[label["record_id"]]
        if (row["record_id"] != label["record_id"] or
                row["class"] != label["kind"] or
                row["episode_id"] != str(plan["episode_id"]) or
                row["scene_id"] != plan["scene_id"] or
                row["task_success_for_audit_only"] != label[
                    "task_success_for_audit_only"] or
                not math.isfinite(row["score"])):
            raise ValueError(f"changed pair score: {pair_id}")
        by_kind[row["class"]].append(row)
        by_record[row["record_id"]][row["class"]] = row
    if seen != set(truth):
        raise ValueError("development score coverage incomplete")
    positive = by_kind["crossing"]
    negative = [x for kind in KINDS[1:] for x in by_kind[kind]]
    candidates = sorted({x["score"] for x in positive + negative}, reverse=True)
    feasible = [(rate(positive, t), -rate(negative, t), t)
                for t in [max(candidates) + 1] + candidates
                if rate(negative, t) <= .05]
    recall, neg_fpr, threshold = max(feasible)
    dev = report["development"]
    for actual, claimed in ((threshold, dev["threshold"]["threshold"]),
                            (recall, dev["threshold"]["pooled_recall"]),
                            (-neg_fpr, dev["threshold"]["pooled_fpr"]),
                            (auc(positive, negative), dev["auc_pooled"])):
        equal(actual, claimed)
    near = [x for x in positive if not x["task_success_for_audit_only"]]
    rates = {"near_failure_recall": rate(near, threshold)}
    for kind in KINDS[1:]:
        rates[kind + "_fpr"] = rate(by_kind[kind], threshold)
    for key, value in rates.items():
        equal(value, dev["rates"][key])
    paired = [(items["crossing"]["score"], items[kind]["score"])
              for items in by_record.values() if "crossing" in items
              for kind in KINDS[1:] if kind in items]
    order = sum(p > n for p, n in paired) / len(paired)
    equal(order, dev["within_route_rank_accuracy"])
    by_contrast = {}
    for kind in KINDS[1:]:
        contrasts = [(items["crossing"]["score"], items[kind]["score"])
                     for items in by_record.values() if "crossing" in items
                     and kind in items]
        by_contrast[kind] = {
            "pairs": len(contrasts),
            "crossing_higher_rate": sum(p > n for p, n in contrasts) /
                                    len(contrasts),
        }
    for scale, key in (("episode_macro", "episode_id"),
                       ("scene_macro", "scene_id")):
        for kind in KINDS:
            count, value = macro(by_kind[kind], key, threshold)
            claimed = dev[scale][kind]
            if count != claimed["groups"]:
                raise ValueError("macro group count mismatch")
            equal(value, claimed["macro_rate"])
    ids = {kind: len({x["episode_id"] for x in by_kind[kind]})
           for kind in KINDS}
    gates = {
        "crossing_episode_ids_at_least_50": ids["crossing"] >= 50,
        "far_episode_ids_at_least_50": ids["far_nonarrival"] >= 50,
        "retreat_episode_ids_at_least_25": ids["retreat"] >= 25,
        "wrong_episode_ids_at_least_30": ids["wrong_instruction"] >= 30,
        "pooled_fpr_at_most_0_05": -neg_fpr <= .05,
        "crossing_recall_at_least_0_55": recall >= .55,
        "near_failure_recall_at_least_0_50": rates[
            "near_failure_recall"] >= .50,
        "retreat_fpr_at_most_0_10": rates["retreat_fpr"] <= .10,
        "wrong_fpr_at_most_0_12": rates["wrong_instruction_fpr"] <= .12,
    }
    if gates != dev["model_gates"]:
        raise ValueError("model gate mismatch")
    gain_text = auc(positive, negative) - text_only["auc_pooled"]
    gain_shuffle = auc(positive, negative) - shuffled["auc_pooled"]
    output = {
        "schema": "multiview_event_independent_development_recount_v1",
        "capture_manifest_sha256": capture_sha,
        "selected_step": report["selected_step"],
        "records": len(plans), "pairs": len(seen),
        "counts": {kind: len(by_kind[kind]) for kind in KINDS},
        "unique_episode_ids": ids,
        "auc_pooled": auc(positive, negative),
        "threshold": threshold, "crossing_recall": recall,
        "pooled_fpr": -neg_fpr, "rates": rates,
        "within_route_rank_accuracy": order,
        "same_record_contrasts": by_contrast,
        "auc_gain_over_text_only": gain_text,
        "auc_gain_over_image_shuffle": gain_shuffle,
        "model_gates": gates,
        "full_gate_passed": all(gates.values()) and gain_text >= .05 and
                            gain_shuffle >= .05,
        "audit_opened": False, "navigation_result": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
