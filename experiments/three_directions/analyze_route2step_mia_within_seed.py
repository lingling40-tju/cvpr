"""Post hoc within-seed restriction of the frozen MIA development screen.

This cannot become a prospective n=4 online-group result: the original
four-record sets were assembled across seeds and contain few such pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re

from analyze_route2step_mia_screen import stage_score, summarize_pairs
from preflight_same_start_pairwise import action_prefix, forward_meters


SEED = re.compile(r"^s(\d+)_")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "record-root", "responses", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "route2step_mia_group4_screen_manifest_v1" or \
            manifest.get("groups") != 16 or manifest.get("queries") != 128:
        raise ValueError("frozen MIA screen changed")
    manifest_sha = digest(args.manifest)
    responses = [json.loads(line) for line in args.responses.read_text().splitlines()]
    by_key = {(row["record_id"], row["anchor"]): row for row in responses}
    if len(responses) != 128 or len(by_key) != 128:
        raise ValueError("incomplete or duplicate response cache")
    groups = []
    for group in manifest["selected"]:
        if len(group["records"]) != 4:
            raise ValueError("four-record group changed")
        members = []
        for rec in group["records"]:
            rid = rec["record_id"]
            path = (args.record_root / "development" / "records" /
                    f"{rid}.json")
            source = json.loads(path.read_text())
            if digest(path) != rec["sha256"] or \
                    source["record_id"] != rid or \
                    source["scene_id"] != group["scene_id"] or \
                    str(source["episode_id"]) != group["episode_id"] or \
                    hashlib.sha256(source["instruction"].encode()).hexdigest() != \
                    group["instruction_sha256"]:
                raise ValueError(f"source mismatch: {rid}")
            match = SEED.match(rid)
            if match is None:
                raise ValueError(f"record seed absent: {rid}")
            scores = {}
            for anchor in (3, 6):
                row = by_key.pop((rid, anchor))
                if row["screen_manifest_sha256"] != manifest_sha or \
                        row["record_sha256"] != rec["sha256"]:
                    raise ValueError("response provenance mismatch")
                scores[anchor], _ = stage_score(source["instruction"],
                                                row["response"])
            members.append((int(match.group(1)), source, scores))
        groups.append((group, members))
    if by_key:
        raise ValueError("unexpected cached responses")
    metrics = {}
    for anchor in (3, 6):
        pairs = []
        for group, members in groups:
            for (ls, left, lscore), (rs, right, rscore) in \
                    itertools.combinations(members, 2):
                if ls != rs:
                    continue
                ld = float(left["turns"][anchor - 1]
                           ["distance_to_goal_for_label_only"])
                rd = float(right["turns"][anchor - 1]
                           ["distance_to_goal_for_label_only"])
                if abs(ld - rd) < 1.0 or \
                        action_prefix(left, anchor) == action_prefix(right, anchor):
                    continue
                lv, rv = lscore[anchor], rscore[anchor]
                non_tie = lv is not None and rv is not None and lv != rv
                mia = float((lv > rv) == (ld < rd)) if non_tie else .5
                forward = forward_meters(left, anchor) - \
                    forward_meters(right, anchor)
                action = float((forward > 0) == (ld < rd)) if forward else .5
                pairs.append((group["scene_id"], group["episode_id"],
                              mia, action, non_tie))
        metrics[str(anchor)] = summarize_pairs(pairs)
    report = {
        "schema": "route2step_mia_within_seed_posthoc_v1",
        "manifest_sha256": manifest_sha,
        "response_cache_sha256": digest(args.responses),
        "anchors": metrics,
        "interpretation": (
            "Post hoc restriction of a reused train-scene development "
            "proxy. The original four-record sets mix policy seeds; "
            "few within-seed pairs remain. No prospective audit, online "
            "n=4 RL, human semantic truth, or val-unseen result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
