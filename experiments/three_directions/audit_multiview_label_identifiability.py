"""CPU-only audit of geometric wrong-instruction labels and observability.

This is a post hoc source audit, not a semantic accuracy estimate or gate.
It reads no RGB frames or reserved audit scenes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row["correct_higher"])
    return {"groups": len(groups), "macro_rate": sum(
        sum(values) / len(values) for values in groups.values()) /
        len(groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("capture", "labels", "development-scores", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    capture, labels, scored = map(read, (args.capture, args.labels,
                                        args.development_scores))
    capture_sha = sha(args.capture)
    if capture["schema"] != "multiview_event_rgb_capture_manifest_v1" or \
            labels["capture_manifest_sha256"] != capture_sha or \
            scored["capture_manifest_sha256"] != capture_sha or \
            capture["group_size"] != 4:
        raise ValueError("changed frozen source")
    sections = {}
    dev_plans = {}
    for part in ("fit", "development"):
        plans = {x["record_id"]: x for x in capture["selected"][part]}
        by_record = defaultdict(dict)
        for label in labels["selected"][part]:
            by_record[label["record_id"]][label["kind"]] = label
        comparable = []
        for rid, parts in by_record.items():
            if "wrong_instruction" not in parts:
                continue
            plan = plans[rid]
            pos, wrong = parts["crossing"], parts["wrong_instruction"]
            if (not plan["wrong_instruction"] or
                    plan["instruction"] == plan["wrong_instruction"] or
                    (pos["before_state_index"], pos["after_state_index"]) !=
                    (wrong["before_state_index"], wrong["after_state_index"]) or
                    pos["before_distance_m_for_audit_only"] != wrong[
                        "before_distance_m_for_audit_only"] or
                    pos["after_distance_m_for_audit_only"] != wrong[
                        "after_distance_m_for_audit_only"]):
                raise ValueError(f"not an identical-state instruction contrast: {rid}")
            comparable.append(plan)
        if part == "development":
            dev_plans = {x["record_id"]: x for x in comparable}
        either_deictic = sum(any(re.search(
            r"\b(here|there|this|that)\b", x[field], re.I)
            for field in ("terminal_clause", "wrong_terminal_clause"))
            for x in comparable)
        either_short = sum(any(len(re.findall(r"[A-Za-z]+", x[field])) < 5
            for field in ("terminal_clause", "wrong_terminal_clause"))
            for x in comparable)
        sections[part] = {
            "identical_state_correct_wrong_pairs": len(comparable),
            "unique_episode_ids": len({str(x["episode_id"]) for x in
                                       comparable}),
            "unique_scenes": len({x["scene_id"] for x in comparable}),
            "identical_terminal_clause_count": sum(
                x["terminal_clause"].casefold() ==
                x["wrong_terminal_clause"].casefold() for x in comparable),
            "either_clause_has_deictic_word_count": either_deictic,
            "either_clause_fewer_than_five_words_count": either_short,
        }
    by_record = defaultdict(dict)
    for row in scored["scores"]:
        if row["record_id"] in dev_plans:
            by_record[row["record_id"]][row["class"]] = row
    contrasts = []
    for rid, plan in dev_plans.items():
        parts = by_record[rid]
        if set(parts) & {"crossing", "wrong_instruction"} != \
                {"crossing", "wrong_instruction"}:
            raise ValueError(f"missing development model score: {rid}")
        contrasts.append({
            "episode_id": str(plan["episode_id"]),
            "scene_id": plan["scene_id"],
            "correct_higher": parts["crossing"]["score"] >
                              parts["wrong_instruction"]["score"],
        })
    output = {
        "schema": "multiview_wrong_instruction_identifiability_audit_v1",
        "capture_manifest_sha256": capture_sha,
        "pair_labels_sha256": sha(args.labels),
        "development_scores_sha256": sha(args.development_scores),
        "source": sections,
        "development_correct_vs_wrong_same_images": {
            "pairs": len(contrasts),
            "correct_higher": sum(x["correct_higher"] for x in contrasts),
            "pooled_rate": sum(x["correct_higher"] for x in contrasts) /
                           len(contrasts),
            "episode_macro": macro(contrasts, "episode_id"),
            "scene_macro": macro(contrasts, "scene_id"),
        },
        "label_basis": "The wrong-instruction class is an alternate-goal geometric counterfactual on identical simulator states, not independently adjudicated semantic visibility.",
        "interpretation": "The score and text counts are post hoc observability diagnostics, not semantic truth, model selection, or a new training gate.",
        "rgb_files_opened": 0,
        "reserved_audit_opened": False,
        "navigation_result": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
