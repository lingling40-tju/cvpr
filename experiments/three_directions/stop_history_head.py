"""Small two-head readout and audited feature loading for STOP/progress."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class StopProgressHead(nn.Module):
    def __init__(self, fit_mean: torch.Tensor, fit_std: torch.Tensor):
        super().__init__()
        if fit_mean.shape != (2048,) or fit_std.shape != (2048,):
            raise ValueError("unexpected fitted feature dimension")
        self.register_buffer("fit_mean", fit_mean.float())
        self.register_buffer("fit_std", fit_std.float().clamp_min(1e-3))
        self.trunk = nn.Sequential(
            nn.Linear(2048, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.stop = nn.Linear(64, 1)
        self.progress = nn.Linear(64, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk((hidden.float() - self.fit_mean) / self.fit_std)
        return self.stop(h).squeeze(-1), torch.tanh(self.progress(h)).squeeze(-1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_part(feature_root: Path, label_audit: dict, part: str,
              manifest_sha: str) -> dict:
    records = []
    scenes = []
    ids = []
    for label in label_audit["labels"][part]:
        if not label["within_12_turns"]:
            continue
        eid = str(label["episode_id"])
        cached = torch.load(feature_root / part / "records" / f"{eid}.pt",
                            map_location="cpu", weights_only=True)
        if (cached["manifest_sha256"] != manifest_sha or
                cached["episode_id"] != eid or
                cached["hidden"].shape != (3, 2048) or
                cached["stop_margin"].shape != (3,) or
                not torch.isfinite(cached["hidden"]).all()):
            raise ValueError(f"invalid cached state {part}/{eid}")
        safe = bool(label["safe_wrong_instruction"])
        if safe and (cached["wrong_hidden"].shape != (2048,) or
                     not torch.isfinite(cached["wrong_hidden"]).all()):
            raise ValueError(f"missing safe wrong state {part}/{eid}")
        ids.append(eid)
        scenes.append(str(label["scene_id"]))
        records.append((cached, label, safe))
    if not records:
        raise ValueError(f"empty feature part {part}")
    hidden = torch.stack([row[0]["hidden"].float() for row in records])
    wrong = torch.stack([row[0]["wrong_hidden"].float() if row[2]
                         else torch.zeros(2048) for row in records])
    distances = torch.tensor([
        [float(row[1]["start_distance_to_goal_m_for_label_only"]),
         float(row[1]["mid_distance_to_goal_m_for_label_only"]),
         float(row[1]["end_distance_to_goal_m_for_label_only"])]
        for row in records], dtype=torch.float32)
    safe_mask = torch.tensor([row[2] for row in records], dtype=torch.bool)
    if not torch.isfinite(distances).all() or \
            not (distances[:, 0] >= 3.5).all() or \
            not (distances[:, 2] <= 3.0).all():
        raise ValueError(f"invalid STOP labels {part}")
    return {"episode_ids": ids, "scenes": scenes, "hidden": hidden,
            "wrong_hidden": wrong, "distances": distances,
            "safe_wrong_mask": safe_mask}


def loss_terms(model: StopProgressHead, batch: dict,
               pos_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    hidden, wrong, distance, safe = (batch[key] for key in
                                      ("hidden", "wrong_hidden", "distances",
                                       "safe_wrong_mask"))
    stop_logits, progress = model(hidden)
    wrong_logits, _ = model(wrong)
    stop_labels = (distance <= 3.0).float()
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits, stop_labels, pos_weight=pos_weight)
    if safe.any():
        stop_loss = stop_loss + F.binary_cross_entropy_with_logits(
            wrong_logits[safe], torch.zeros_like(wrong_logits[safe]))
        grounding = F.softplus(1.0 -
                               (stop_logits[safe, 2] - wrong_logits[safe])).mean()
    else:
        grounding = stop_logits.sum() * 0.0
    progress_target = (1.0 - distance / distance[:, :1]).clamp(-1.0, 1.0)
    regression = F.smooth_l1_loss(progress, progress_target)
    gaps = distance[:, :-1] - distance[:, 1:]
    valid = gaps.abs() >= 1.0
    signed_deltas = torch.sign(gaps) * (progress[:, 1:] - progress[:, :-1])
    ordinal = F.softplus(-signed_deltas[valid]).mean() if valid.any() \
        else progress.sum() * 0.0
    return {"stop": stop_loss, "progress": regression,
            "grounding": grounding, "ordinal": ordinal,
            "total": stop_loss + 0.5 * regression +
            0.5 * grounding + 0.25 * ordinal}


def auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    if not len(positive) or not len(negative):
        raise ValueError("empty AUC class")
    comparison = positive[:, None] - negative[None, :]
    return float(((comparison > 0).float() +
                  0.5 * (comparison == 0).float()).mean())


def metrics(model: StopProgressHead, data: dict) -> dict:
    model.eval()
    with torch.inference_mode():
        stop, progress = model(data["hidden"])
        wrong, _ = model(data["wrong_hidden"])
    distance = data["distances"]
    safe = data["safe_wrong_mask"]
    positive = stop[distance <= 3.0]
    negative = torch.cat([stop[distance > 3.0], wrong[safe]])
    gaps = distance[:, :-1] - distance[:, 1:]
    valid = gaps.abs() >= 1.0
    correct = torch.sign(gaps) * (progress[:, 1:] - progress[:, :-1]) > 0
    return {
        "stop_auc": auc(positive, negative),
        "stop_positive": len(positive), "stop_negative": len(negative),
        "progress_pair_accuracy": float(correct[valid].float().mean()),
        "progress_pairs": int(valid.sum()),
        "instruction_swap_accuracy": float((stop[safe, 2] >
                                              wrong[safe]).float().mean()),
        "instruction_swap_pairs": int(safe.sum()),
        "positive_logits": positive.tolist(),
        "negative_logits": negative.tolist(),
    }


def choose_threshold(positive: list[float], negative: list[float]) -> dict:
    candidates = sorted(set(positive + negative), reverse=True)
    candidates = [max(candidates) + 1.0] + candidates + [min(candidates) - 1.0]
    feasible = []
    for threshold in candidates:
        fpr = sum(value >= threshold for value in negative) / len(negative)
        recall = sum(value >= threshold for value in positive) / len(positive)
        if fpr <= 0.10:
            feasible.append((recall, -fpr, threshold))
    recall, negative_fpr, threshold = max(feasible)
    return {"threshold": threshold, "development_fpr": -negative_fpr,
            "development_recall": recall}
