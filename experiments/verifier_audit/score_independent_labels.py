"""Score two independent reviews and an adjudicated label set on selected cases."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABELS = "YNU"


def read_labels(path, expected):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["item_id"] for row in rows]
    if len(ids) != len(expected) or set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: expected exactly {len(expected)} unique item IDs")
    result = {row["item_id"]: row["label"].strip().upper() for row in rows}
    bad = {item: label for item, label in result.items() if label not in LABELS}
    if bad:
        raise ValueError(f"{path}: missing or invalid labels: {bad}")
    return result


def confusion(gold, pred, items):
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for item in items:
        matrix[gold[item]][pred[item]] += 1
    return matrix


def agreement(left, right, items):
    return sum(left[item] == right[item] for item in items) / len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer1", type=Path)
    parser.add_argument("reviewer2", type=Path)
    parser.add_argument("adjudicated", type=Path)
    args = parser.parse_args()
    bundle = json.loads((ROOT / "bundle.json").read_text())
    items = [hashlib.sha256(
        f"{r['episode_id']}:{r['event_index']}:{r['turn']}".encode()
    ).hexdigest()[:12] for r in bundle["records"]]
    expected = set(items)
    if len(expected) != len(items):
        raise ValueError("bundle has duplicate item IDs")
    h1 = read_labels(args.reviewer1, expected)
    h2 = read_labels(args.reviewer2, expected)
    gold = read_labels(args.adjudicated, expected)
    verdicts = json.loads((ROOT / "verdicts.json").read_text())
    prediction = {
        hashlib.sha256(
            f"{r['episode_id']}:{r['event_index']}:{r['turn']}".encode()
        ).hexdigest()[:12]: r["prediction"] for r in verdicts
    }
    if set(prediction) != expected:
        raise ValueError("verifier verdict IDs do not match review package")
    matrix = confusion(gold, prediction, items)
    tp = matrix["Y"]["Y"]
    pred_y = sum(matrix[g]["Y"] for g in LABELS)
    gold_y = sum(matrix["Y"].values())
    result = {
        "selected_cases": len(items),
        "reviewer_agreement": agreement(h1, h2, items),
        "reviewer_disagreement_ids": [item for item in items if h1[item] != h2[item]],
        "adjudicated_label_counts": dict(collections.Counter(gold.values())),
        "verifier_confusion": matrix,
        "verifier_three_way_agreement": agreement(gold, prediction, items),
        "verifier_completion_precision": tp / pred_y if pred_y else None,
        "verifier_completion_recall": tp / gold_y if gold_y else None,
        "note": "Selected transitions; descriptive statistics, not population accuracy.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
