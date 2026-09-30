"""Compare the frozen semantic verifier with previously frozen blind labels."""

import base64
import collections
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:5003/verify"
STATUS = {"completed": "Y", "not_completed": "N", "uncertain": "U"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query(record):
    images = {}
    for field in ("before", "after"):
        source = ROOT / record["images"][field]["path"]
        assert sha256(source) == record["images"][field]["sha256"]
        images[field] = base64.b64encode(source.read_bytes()).decode()
    body = {
        "event": record["event"],
        "before": images["before"],
        "after": images["after"],
        "actions": record["actions"],
        "displacement": record["displacement"],
        "vertical_delta": record["vertical_delta"],
    }
    request = Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result)
    return result


def main():
    bundle = json.loads((ROOT / "bundle.json").read_text())
    assert sha256(ROOT / "annotations.csv") == bundle["annotation_csv_sha256"]
    output = ROOT / "verdicts.json"
    if output.exists():
        verdicts = json.loads(output.read_text())
        assert len(verdicts) <= len(bundle["records"])
    else:
        verdicts = []
    for record in bundle["records"][len(verdicts):]:
        result = query(record)
        verdicts.append({
            "episode_id": record["episode_id"],
            "event_index": record["event_index"],
            "turn": record["turn"],
            "gold": record["gold"],
            "prediction": STATUS[result["status"]],
            "response": result,
        })
        output.write_text(json.dumps(verdicts, indent=2) + "\n")
        print(len(verdicts), record["episode_id"], record["turn"],
              record["gold"], verdicts[-1]["prediction"], flush=True)
    confusion = {gold: {pred: 0 for pred in "YNU"} for gold in "YNU"}
    for verdict in verdicts:
        confusion[verdict["gold"]][verdict["prediction"]] += 1
    tp = confusion["Y"]["Y"]
    predicted_y = sum(confusion[gold]["Y"] for gold in "YNU")
    gold_y = sum(confusion["Y"].values())
    summary = {
        "count": len(verdicts),
        "episode_count": len({x["episode_id"] for x in verdicts}),
        "label_counts": dict(collections.Counter(x["gold"] for x in verdicts)),
        "prediction_counts": dict(collections.Counter(x["prediction"] for x in verdicts)),
        "confusion": confusion,
        "three_way_accuracy": sum(confusion[status][status] for status in "YNU") / len(verdicts),
        "completed_precision": tp / predicted_y if predicted_y else None,
        "completed_recall": tp / gold_y if gold_y else None,
        "false_positive_count_on_clear_negatives": confusion["N"]["Y"],
        "annotation_csv_sha256": bundle["annotation_csv_sha256"],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
