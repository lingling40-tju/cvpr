"""Replay frozen blind examples against a verifier without changing labels."""

import argparse
import base64
import collections
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


STATUS = {"completed": "Y", "not_completed": "N", "uncertain": "U"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:5004/verify")
    parser.add_argument("--resize-long-edge", type=int, default=448)
    args = parser.parse_args()
    bundle = json.loads((args.bundle_dir / "bundle.json").read_text())
    assert digest(args.bundle_dir / "annotations.csv") == bundle["annotation_csv_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "verdicts.json"
    verdicts = json.loads(output.read_text()) if output.exists() else []
    assert len(verdicts) <= len(bundle["records"])
    for record in bundle["records"][len(verdicts):]:
        images = {}
        for field in ("before", "after"):
            entry = record["images"][field]
            image = args.bundle_dir / entry["path"]
            assert digest(image) == entry["sha256"]
            frame = Image.open(image).convert("RGB")
            frame.thumbnail((args.resize_long_edge, args.resize_long_edge))
            stream = io.BytesIO()
            frame.save(stream, format="JPEG", quality=78)
            images[field] = base64.b64encode(stream.getvalue()).decode()
        body = {
            "event": record["event"], "before": images["before"], "after": images["after"],
            "actions": record["actions"], "displacement": record["displacement"],
            "vertical_delta": record["vertical_delta"],
        }
        request = Request(args.url, json.dumps(body).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request, timeout=150) as response:
            result = json.load(response)
        assert "error" not in result, result
        verdicts.append({
            "episode_id": record["episode_id"], "event_index": record["event_index"],
            "turn": record["turn"], "gold": record["gold"],
            "prediction": STATUS[result["status"]], "response": result,
        })
        output.write_text(json.dumps(verdicts, indent=2) + "\n")
        print(len(verdicts), record["episode_id"], record["turn"],
              record["gold"], verdicts[-1]["prediction"], flush=True)
    confusion = {gold: {pred: 0 for pred in "YNU"} for gold in "YNU"}
    for verdict in verdicts:
        confusion[verdict["gold"]][verdict["prediction"]] += 1
    summary = {
        "count": len(verdicts), "episode_count": len({v["episode_id"] for v in verdicts}),
        "label_counts": dict(collections.Counter(v["gold"] for v in verdicts)),
        "prediction_counts": dict(collections.Counter(v["prediction"] for v in verdicts)),
        "confusion": confusion,
        "three_way_agreement_with_single_ai": sum(confusion[x][x] for x in "YNU") / len(verdicts),
        "completed_predictions_matching_single_ai": confusion["Y"]["Y"],
        "completed_prediction_count": sum(confusion[x]["Y"] for x in "YNU"),
        "completed_on_clear_ai_negative": confusion["N"]["Y"],
        "annotation_csv_sha256": bundle["annotation_csv_sha256"],
        "label_source": "one AI reviewer; no independent human ground truth",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
