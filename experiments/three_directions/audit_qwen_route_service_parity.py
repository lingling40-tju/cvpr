"""Check live Qwen reward service against frozen cached-route margins.

Sends original cached JPEG bytes, with no recompression, for the ten
fit groups that overlap the selected online train rows. This audits
service weights, prompt, A/B averaging, route order, and ID mapping.
It does not test fresh online simulator images or navigation outcomes.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-manifest", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8031")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    online = json.loads(args.online_manifest.read_text())
    policy = json.loads(args.policy_manifest.read_text())
    if online["schema"] != "qwen3_group4_exact_start_dataset_v1" or \
            policy["schema"] != "qwen3_policy_route_manifest_v1":
        raise ValueError("manifest schema mismatch")
    selected = {row["episode_id"]: row for row in online["rows"]}
    fit = {group["episode_id"]: group for group in policy["selected"]["fit"]
           if group["episode_id"] in selected}
    if len(fit) != 10 or any(
            group["wrong_episode_id"] != selected[eid]["wrong_episode_id"]
            for eid, group in fit.items()):
        raise ValueError("selected exact-start overlap changed")
    scored = {}
    for shard_index in range(4):
        shard = json.loads((args.shard_root /
                            f"terminal_fit_shard{shard_index}.json").read_text())
        if shard["schema"] != "qwen3_policy_terminal_shard_v1" or \
                shard["part"] != "fit":
            raise ValueError("frozen score shard mismatch")
        for group in shard["groups"]:
            if group["episode_id"] in fit:
                scored[group["episode_id"]] = group
    if set(scored) != set(fit):
        raise ValueError("frozen fit score coverage mismatch")
    health_before = requests.get(args.url + "/health", timeout=5).json()
    if health_before["status"] != "ok" or \
            health_before["variant"] != "qwen3_exact_start_group_rank_v1":
        raise ValueError("wrong live service")
    rows = []
    for eid in sorted(fit):
        for route in scored[eid]["routes"]:
            rid = route["record_id"]
            if rid not in {item["record_id"] for item in fit[eid]["routes"]}:
                raise ValueError("route ID mismatch")
            record = json.loads((args.replay_root / "records" /
                                 f"{rid}.json").read_text())
            if str(record["episode_id"]) != eid:
                raise ValueError("replay episode mismatch")
            paths = [record["initial_image"]] + \
                    [turn["image"] for turn in record["turns"]]
            turn = len(record["turns"])
            indices = [math.floor(i * turn / 5 + .5) for i in range(6)]
            if route["states"][-1]["sampled_frame_indices"] != indices:
                raise ValueError("frozen sampling rule mismatch")
            images = [base64.b64encode((args.replay_root / paths[i]).read_bytes()).decode()
                      for i in indices]
            response = requests.post(args.url + "/score", json={
                "episode_id": eid, "instruction": record["instruction"],
                "images": images}, timeout=120)
            response.raise_for_status()
            answer = response.json()
            if answer["status"] != "ok" or answer["scored_views"] != 6:
                raise ValueError("live score failed")
            expected = float(route["states"][-1]["correct_margin_average"])
            actual = float(answer["raw"])
            rows.append({"record_id": rid, "episode_id": eid,
                         "offline_margin": expected, "live_margin": actual,
                         "absolute_difference": abs(actual - expected)})
    health_after = requests.get(args.url + "/health", timeout=5).json()
    if health_after["requests"] - health_before["requests"] != len(rows) or \
            len(rows) != 40 or max(row["absolute_difference"] for row in rows) > 1e-5:
        raise ValueError("live and frozen route margins differ")
    report = {"schema": "qwen3_route_service_parity_v1",
              "overlap_groups": len(fit), "routes": len(rows),
              "max_absolute_margin_difference": max(
                  row["absolute_difference"] for row in rows),
              "request_delta": health_after["requests"] - health_before["requests"],
              "interpretation": "Original cached JPEG parity only; fresh simulator images and navigation performance unverified.",
              "routes_audit": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("overlap_groups", "routes", "max_absolute_margin_difference",
                       "request_delta")}, indent=2))


if __name__ == "__main__":
    main()
