"""Fit one predeclared two-head readout on train-scene history features.

Checkpoint/epoch and STOP threshold are chosen using development scenes.
This script never loads audit features, labels, or val-unseen episodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from stop_history_head import (StopProgressHead, choose_threshold, digest,
                               load_part, loss_terms, metrics)


def take(data: dict, indices: torch.Tensor) -> dict:
    return {key: value[indices] for key, value in data.items()
            if isinstance(value, torch.Tensor)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.seed != 11 or args.epochs != 50 or args.batch_size != 32:
        raise ValueError("fixed pilot readout configuration changed")
    manifest_sha = digest(args.manifest)
    labels = json.loads(args.label_audit.read_text())
    if labels["manifest_sha256"] != manifest_sha:
        raise ValueError("label/manifest mismatch")
    fit_summary = json.loads((args.features_root / "fit" / "summary.json").read_text())
    development_summary = json.loads((args.features_root / "development" / "summary.json").read_text())
    for part, summary in (("fit", fit_summary), ("development", development_summary)):
        expected = sum(row["within_12_turns"] for row in labels["labels"][part])
        if (summary["manifest_sha256"] != manifest_sha or
                summary["requested"] != summary["completed"] or
                summary["completed"] != expected or
                summary["errors"] or summary["limit"] != 0):
            raise ValueError(f"incomplete {part} features")
    begun = time.time()
    fit = load_part(args.features_root, labels, "fit", manifest_sha)
    development = load_part(args.features_root, labels, "development", manifest_sha)
    if len(fit["episode_ids"]) != 692 or len(development["episode_ids"]) != 112:
        raise ValueError("unexpected fixed train/development coverage")
    fit_vectors = torch.cat([
        fit["hidden"].reshape(-1, 2048),
        fit["wrong_hidden"][fit["safe_wrong_mask"]],
    ])
    mean = fit_vectors.mean(dim=0)
    std = fit_vectors.std(dim=0).clamp_min(1e-3)
    positive = int((fit["distances"] <= 3.0).sum())
    negative = int((fit["distances"] > 3.0).sum() + fit["safe_wrong_mask"].sum())
    pos_weight = torch.tensor(negative / positive)
    scene_counts = Counter(fit["scenes"])
    weights = torch.tensor([1.0 / scene_counts[scene] for scene in fit["scenes"]])
    weights /= weights.sum()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = StopProgressHead(mean, std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sample_indices = torch.multinomial(weights, len(weights),
                                           replacement=True, generator=generator)
        for indices in sample_indices.split(args.batch_size):
            batch = take(fit, indices)
            terms = loss_terms(model, batch, pos_weight)
            if not torch.isfinite(terms["total"]):
                raise ValueError(f"nonfinite fit loss epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_terms = loss_terms(model, development, pos_weight)
            dev_loss = float(dev_terms["total"])
        history.append({"epoch": epoch, "development_loss": dev_loss})
        if dev_loss < best_loss - 1e-5:
            best_loss, best_epoch = dev_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 8:
            break
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch={epoch} development_loss={dev_loss:.4f} best={best_epoch}", flush=True)
    if best_state is None:
        raise ValueError("no selected readout")
    model.load_state_dict(best_state)
    report = metrics(model, development)
    threshold = choose_threshold(report["positive_logits"], report["negative_logits"])
    report.pop("positive_logits")
    report.pop("negative_logits")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "stop_history_two_head_seed11_v1",
                "model_state": best_state,
                "manifest_sha256": manifest_sha,
                "label_audit_sha256": digest(args.label_audit),
                "fit_summary_sha256": digest(args.features_root / "fit" / "summary.json"),
                "development_summary_sha256": digest(args.features_root / "development" / "summary.json"),
                "selected_epoch": best_epoch, "seed": args.seed,
                "development_stop_threshold": threshold["threshold"]},
               output / "head.pt")
    summary = {
        "schema": "stop_history_two_head_fit_v1",
        "manifest_sha256": manifest_sha,
        "seed": args.seed, "selected_epoch": best_epoch,
        "max_epochs": args.epochs, "batch_size_trajectories": args.batch_size,
        "fit_trajectories": len(fit["episode_ids"]),
        "development_trajectories": len(development["episode_ids"]),
        "fit_positive_labels": positive, "fit_negative_labels": negative,
        "checkpoint_selection": "lowest development composite loss; no audit read",
        "development_metrics": report,
        "development_stop_threshold": threshold,
        "development_loss_history": history,
        "elapsed_seconds": time.time() - begun,
    }
    (output / "fit_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("selected_epoch", "development_metrics",
                       "development_stop_threshold", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
