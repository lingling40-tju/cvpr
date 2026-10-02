"""Compare live failure-only reward responses with cached SFT features."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import requests
import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8024")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cache = torch.load(args.features, map_location="cpu", weights_only=False)
    calibration = json.loads(args.calibration.read_text())
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    if cache["manifest_sha256"] != digest(args.manifest) or \
            cache["hidden"].shape != (2 * len(manifest["pairs"]), 4, 2048) or \
            calibration["new_encoder_sha256"] != digest(args.encoder) or \
            checkpoint["manifest_sha256"] != digest(args.manifest):
        raise ValueError("parity input provenance mismatch")
    health = requests.get(args.url.rstrip("/") + "/health", timeout=3).json()
    if health["status"] != "ok" or \
            health["reward_variant"] != "failure_only_temporal_v1" or \
            health["encoder_sha256"] != digest(args.encoder) or \
            health["calibration_sha256"] != digest(args.calibration):
        raise ValueError("wrong live reward service")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TemporalPotential().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    rows = []
    for index, pair in enumerate(manifest["pairs"][:args.pairs]):
        for role_index, role in enumerate(("near", "far")):
            record_id = pair["pair_id"] + "_" + role
            if cache["record_ids"][2 * index + role_index] != record_id:
                raise ValueError("feature identity mismatch")
            record = json.loads((args.replay_root / "records" /
                                 f"{record_id}.json").read_text())
            if record["record_id"] != record_id or len(record["frames"]) != 4:
                raise ValueError("replay identity mismatch")
            payload = {"instruction": pair["instruction"],
                       "images": [base64.b64encode((args.replay_root / frame["image"])
                                                   .read_bytes()).decode()
                                  for frame in record["frames"]],
                       "initial": [bool(frame["initial"]) for frame in record["frames"]]}
            response = requests.post(args.url.rstrip("/") + "/score",
                                     json=payload, timeout=120)
            response.raise_for_status()
            live = response.json()
            if live["status"] != "ok":
                raise ValueError("service scoring failed")
            hidden = F.normalize(cache["hidden"][2 * index + role_index]
                                 .float(), dim=-1).unsqueeze(0)
            cached_score = float(predict(model, hidden, device)[0, -1])
            cached_raw = cached_score / calibration["temporal_scale"]
            cached_bonus = 1 / (1 + math.exp(-cached_raw))
            rows.append({"record_id": record_id,
                         "temporal_abs_error": abs(live["temporal_score"] - cached_score),
                         "bonus_abs_error": abs(live["bonus"] - cached_bonus)})
    if len(rows) != 2 * args.pairs or \
            max(row["temporal_abs_error"] for row in rows) > 1e-3 or \
            max(row["bonus_abs_error"] for row in rows) > 1e-3:
        raise ValueError("online/offline failure reward parity failed")
    result = {"schema": "failure_only_service_parity_v1",
              "interpretation": "Fit-scene service wiring only; no RL or val-unseen result.",
              "manifest_sha256": digest(args.manifest),
              "encoder_sha256": digest(args.encoder),
              "calibration_sha256": digest(args.calibration),
              "pairs": args.pairs, "records": len(rows),
              "max_temporal_abs_error": max(row["temporal_abs_error"] for row in rows),
              "max_bonus_abs_error": max(row["bonus_abs_error"] for row in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
