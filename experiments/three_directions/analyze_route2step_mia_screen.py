"""Frozen offline gate for Route2Step MIA stage as a group-four reward cue.

This is a reused R2R-train scene-disjoint development screen. Simulator
distance is read only to score the cached prediction, never as model input.
Ties and unaligned answers score 0.5, so abstention cannot inflate accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

from preflight_same_start_pairwise import action_prefix, forward_meters


WORDS = re.compile(r"[a-z0-9]+")
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
CONTENT_STOPWORDS = {"the", "a", "an", "and", "to", "of", "in", "on", "at",
                     "through", "past", "by", "you", "your", "are", "is"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text: str) -> list[str]:
    return WORDS.findall(text.lower())


def stage_score(instruction: str, response: str) -> tuple[float | None, str]:
    match = ANSWER.search(response)
    if not match:
        return None, "missing_answer_tag"
    answer = tokens(match.group(1))
    source = tokens(instruction)
    if not answer or not source or answer == ["stop"] or answer == ["none"]:
        return None, "stop_or_empty"
    answer_counts = Counter(answer)
    best = (-1.0, 0, 0, 0)
    max_span = min(len(source), len(answer) + 3)
    for start in range(len(source)):
        for end in range(start + 1, min(len(source), start + max_span) + 1):
            span = source[start:end]
            overlap = sum((Counter(span) & answer_counts).values())
            precision = overlap / len(span)
            recall = overlap / len(answer)
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            content = sum((Counter(w for w in span if w not in CONTENT_STOPWORDS)
                           & Counter(w for w in answer if w not in CONTENT_STOPWORDS)).values())
            candidate = (f1, content, -start, end)
            if candidate > best:
                best = candidate
    f1, content, negative_start, end = best
    if f1 < 0.55 or content < 2:
        return None, "unaligned_answer"
    start = -negative_start
    return (start + end) / (2 * len(source)), "aligned"


def summarize_pairs(rows: list[tuple[str, str, float, float, bool]]) -> dict:
    # scene, episode, MIA correctness, action-baseline correctness, non-tie
    episode = defaultdict(list)
    hard = defaultdict(list)
    for scene, eid, mia, action, non_tie in rows:
        episode[(scene, eid)].append((mia, action, non_tie))
        if action == 0.0:
            hard[(scene, eid)].append(mia)
    scene_scores = defaultdict(list)
    scene_baselines = defaultdict(list)
    for (scene, _), values in episode.items():
        scene_scores[scene].append(statistics.mean(v[0] for v in values))
        scene_baselines[scene].append(statistics.mean(v[1] for v in values))
    return {
        "qualifying_pairs": len(rows), "episode_groups": len(episode),
        "scenes": len(scene_scores),
        "mia_pair_accuracy": statistics.mean(v[2] for v in rows) if rows else None,
        "mia_episode_macro": statistics.mean(
            statistics.mean(v[0] for v in values) for values in episode.values())
        if episode else None,
        "mia_scene_macro": statistics.mean(
            statistics.mean(v) for v in scene_scores.values())
        if scene_scores else None,
        "action_episode_macro": statistics.mean(
            statistics.mean(v[1] for v in values) for values in episode.values())
        if episode else None,
        "action_scene_macro": statistics.mean(
            statistics.mean(v) for v in scene_baselines.values())
        if scene_baselines else None,
        "non_tie_fraction": statistics.mean(v[4] for v in rows) if rows else None,
        "hard_pairs": sum(len(v) for v in hard.values()),
        "hard_episode_groups": len(hard),
        "hard_episode_macro": statistics.mean(
            statistics.mean(v) for v in hard.values()) if hard else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "record-root", "responses", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.manifest.read_text())
    if frozen["schema"] != "route2step_mia_group4_screen_manifest_v1" or \
            frozen["groups"] != 16 or frozen["queries"] != 128:
        raise ValueError("unexpected frozen screen")
    response_rows = [json.loads(line) for line in args.responses.read_text().splitlines()]
    by_key = {(r["record_id"], r["anchor"]): r for r in response_rows}
    if len(response_rows) != 128 or len(by_key) != 128:
        raise ValueError("incomplete or duplicate MIA responses")
    groups = []
    parse_counts = Counter()
    temporal = []
    all_elapsed = []
    for group in frozen["selected"]:
        records = []
        if len(group["records"]) != 4:
            raise ValueError("group size changed")
        for rec in group["records"]:
            rid = rec["record_id"]
            path = args.record_root / "development" / "records" / f"{rid}.json"
            source = json.loads(path.read_text())
            if digest(path) != rec["sha256"] or \
                    hashlib.sha256(source["instruction"].encode()).hexdigest() != group["instruction_sha256"]:
                raise ValueError(f"source record changed: {rid}")
            scores = {}
            for anchor in (3, 6):
                row = by_key[(rid, anchor)]
                if row["screen_manifest_sha256"] != digest(args.manifest) or \
                        row["record_sha256"] != rec["sha256"]:
                    raise ValueError("response provenance mismatch")
                score, reason = stage_score(source["instruction"], row["response"])
                parse_counts[reason] += 1
                scores[anchor] = score
                all_elapsed.append(float(row["elapsed_seconds"]))
            if scores[3] is not None and scores[6] is not None:
                temporal.append(scores[6] >= scores[3])
            records.append((source, scores))
        groups.append((group, records))
    summaries = {}
    for anchor in (3, 6):
        pairs = []
        for group, records in groups:
            for (left, ls), (right, rs) in itertools.combinations(records, 2):
                ld = float(left["turns"][anchor - 1]["distance_to_goal_for_label_only"])
                rd = float(right["turns"][anchor - 1]["distance_to_goal_for_label_only"])
                if not math.isfinite(ld) or not math.isfinite(rd):
                    raise ValueError("invalid target distance")
                if abs(ld - rd) < 1.0 or \
                        action_prefix(left, anchor) == action_prefix(right, anchor):
                    continue
                lscore, rscore = ls[anchor], rs[anchor]
                non_tie = lscore is not None and rscore is not None and lscore != rscore
                mia = (float((lscore > rscore) == (ld < rd)) if non_tie else 0.5)
                forward = forward_meters(left, anchor) - forward_meters(right, anchor)
                action = (float((forward > 0) == (ld < rd)) if forward else 0.5)
                pairs.append((group["scene_id"], group["episode_id"],
                              mia, action, non_tie))
        summaries[str(anchor)] = summarize_pairs(pairs)
    tag_fraction = 1 - parse_counts["missing_answer_tag"] / 128
    alignment_fraction = parse_counts["aligned"] / 128
    checks = {"answer_tag_at_least_90pct": tag_fraction >= 0.90,
              "alignment_at_least_80pct": alignment_fraction >= 0.80}
    for anchor in ("3", "6"):
        s = summaries[anchor]
        checks[f"anchor{anchor}_at_least_8_groups"] = s["episode_groups"] >= 8
        checks[f"anchor{anchor}_non_tie_at_least_25pct"] = s["non_tie_fraction"] >= 0.25
        checks[f"anchor{anchor}_episode_macro_at_least_72pct"] = s["mia_episode_macro"] >= 0.72
        checks[f"anchor{anchor}_above_action_5pp"] = (
            s["mia_episode_macro"] >= s["action_episode_macro"] + 0.05)
        checks[f"anchor{anchor}_hard_8_groups_60pct"] = (
            s["hard_episode_groups"] >= 8 and s["hard_episode_macro"] >= 0.60)
    report = {
        "schema": "route2step_mia_group4_reused_development_v1",
        "manifest_sha256": digest(args.manifest),
        "response_cache_sha256": digest(args.responses),
        "groups": 16, "queries": 128, "anchors": summaries,
        "answer_tag_fraction": tag_fraction,
        "instruction_span_alignment_fraction": alignment_fraction,
        "parse_counts": dict(parse_counts),
        "temporal_nondecrease_fraction": statistics.mean(temporal) if temporal else None,
        "temporal_eligible_records": len(temporal),
        "sum_inference_seconds": sum(all_elapsed),
        "checks": checks, "passed_exploratory_gate": all(checks.values()),
        "interpretation": "Reused R2R-train scene-disjoint development proxy only; no independent human semantic labels, reward fit, policy RL, or val-unseen navigation result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed_exploratory_gate": report["passed_exploratory_gate"],
                      "checks": checks, "anchors": summaries}))


if __name__ == "__main__":
    main()
