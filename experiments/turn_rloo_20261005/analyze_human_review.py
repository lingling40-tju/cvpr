"""Summarize two locked blind-review CSVs without using a private answer key.

This descriptive report does not adjudicate disagreements or estimate
verifier accuracy. The selected 49 cases are enriched for transitions.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import zipfile


FIELDS = ("reviewer_id", "blind_id", "label_Y_N_U", "confidence_1_3", "brief_evidence")


def read_review(path: Path, expected_ids: set[str]) -> tuple[str, dict[str, dict]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: CSV header mismatch")
        rows = list(reader)
    ids = [r["blind_id"] for r in rows]
    if len(rows) != len(expected_ids) or set(ids) != expected_ids or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: missing or duplicate blind IDs")
    reviewers = {r["reviewer_id"].strip() for r in rows}
    if len(reviewers) != 1 or not next(iter(reviewers)):
        raise ValueError(f"{path}: expected one nonempty reviewer ID")
    for r in rows:
        if r["label_Y_N_U"] not in "YNU" or r["confidence_1_3"] not in "123" or not r["brief_evidence"].strip():
            raise ValueError(f"{path}: incomplete or invalid label for {r['blind_id']}")
    return next(iter(reviewers)), {r["blind_id"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    ap.add_argument("reviewer1", type=Path)
    ap.add_argument("reviewer2", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with zipfile.ZipFile(args.package) as z:
        items = json.loads(z.read("items.json"))
    by_id = {i["blind_id"]: i for i in items}
    if len(items) != 49 or len(by_id) != 49:
        raise ValueError("unexpected review package")
    id1, a = read_review(args.reviewer1, set(by_id))
    id2, b = read_review(args.reviewer2, set(by_id))
    if id1 == id2:
        raise ValueError("reviewer IDs must differ")
    order = [i["blind_id"] for i in items]
    confusion = {x: {y: 0 for y in "YNU"} for x in "YNU"}
    by_type = collections.defaultdict(lambda: {"n": 0, "agreement": 0})
    for bid in order:
        x, y = a[bid]["label_Y_N_U"], b[bid]["label_Y_N_U"]
        confusion[x][y] += 1
        typ = by_id[bid]["event_type"]
        by_type[typ]["n"] += 1
        by_type[typ]["agreement"] += int(x == y)
    n = len(order)
    observed = sum(confusion[x][x] for x in "YNU") / n
    counts1 = {x: sum(confusion[x].values()) for x in "YNU"}
    counts2 = {y: sum(confusion[x][y] for x in "YNU") for y in "YNU"}
    expected = sum(counts1[x] * counts2[x] for x in "YNU") / n**2
    disagreement = [bid for bid in order if a[bid]["label_Y_N_U"] != b[bid]["label_Y_N_U"]]
    result = {
        "schema": "eventtrace_two_human_blind_review_agreement_v1",
        "source_sha256": {
            "answer_free_review_package": hashlib.sha256(args.package.read_bytes()).hexdigest(),
            id1: hashlib.sha256(args.reviewer1.read_bytes()).hexdigest(),
            id2: hashlib.sha256(args.reviewer2.read_bytes()).hexdigest(),
        },
        "selected_transition_enriched_cases": n,
        "independent_reviewers_user_confirmed": True,
        "label_counts": {id1: counts1, id2: counts2},
        "cross_tab_rows_reviewer1_columns_reviewer2": confusion,
        "exact_agreement": sum(confusion[x][x] for x in "YNU"),
        "exact_agreement_fraction": observed,
        "cohen_kappa": (observed - expected) / (1 - expected),
        "by_event_type": dict(by_type),
        "disagreement_blind_ids": disagreement,
        "adjudicated": False,
        "model_accuracy_estimated": False,
        "interpretation": "Selected 49-case rubric pilot. No model confusion or population precision without independent adjudication and sampling.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cases": n, "agreement": result["exact_agreement"],
                      "kappa": result["cohen_kappa"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
