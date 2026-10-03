"""Fixed low-capacity on-policy STOP-readiness calibration screen.

Labels are simulator distance on *train* episodes. Scene-disjoint head
evaluation is exploratory because the frozen encoders saw related train data.
No val-unseen result or online policy update is inferred from this probe.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


FEATURES = (
    "visual_terminal", "visual_change", "sft_stop_terminal",
    "sft_stop_change", "sft_hidden_change_norm",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(positive, negative):
    if len(positive) == 0 or len(negative) == 0:
        return None
    pairs = positive[:, None] - negative[None, :]
    return float((np.count_nonzero(pairs > 0) +
                  0.5 * np.count_nonzero(pairs == 0)) / pairs.size)


def metrics(records, scores, indices, threshold):
    selected = np.asarray(indices)
    near = np.asarray([records[i]["terminal_distance_m_for_replay_audit_only"]
                       <= 3.5 for i in selected])
    failed = np.asarray([not records[i]["task_success"] for i in selected])
    far = ~near
    values = scores[selected]
    near_failed = near & failed
    return {
        "records": len(selected),
        "scenes": len({records[i]["scene_id"] for i in selected}),
        "near_or_success": int(near.sum()),
        "near_failed_stops": int(near_failed.sum()),
        "far_failed_stops": int(far.sum()),
        "near_vs_far_auc": auc(values[near], values[far]),
        "near_failed_vs_far_failed_auc": auc(values[near_failed], values[far]),
        "far_failed_positive_rate_at_fit_threshold": float((values[far] >= threshold).mean()),
        "near_or_success_recall_at_fit_threshold": float((values[near] >= threshold).mean()),
        "successful_stop_recall_at_fit_threshold": float(
            (values[~failed] >= threshold).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sft-features", type=Path, required=True)
    parser.add_argument("--visual-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    visual = json.loads(args.visual_scores.read_text())
    sft = torch.load(args.sft_features, map_location="cpu", weights_only=True)
    ids = [row["record_id"] for row in records]
    assert len(ids) == 531
    assert visual["manifest_sha256"] == sft["manifest_sha256"] == digest(args.manifest)
    assert [row["record_id"] for row in visual["records"]] == ids == sft["record_ids"]
    hidden = F.normalize(sft["hidden"].float(), dim=-1)
    stop = sft["stop_margin"].float().numpy()
    v0 = np.asarray([row["initial_score"] for row in visual["records"]])
    v1 = np.asarray([row["terminal_score"] for row in visual["records"]])
    data = np.stack([
        v1, v1 - v0, stop[:, 1], stop[:, 1] - stop[:, 0],
        (hidden[:, 1] - hidden[:, 0]).norm(dim=-1).numpy(),
    ], axis=1)
    scene_order = sorted({row["scene_id"] for row in records},
                         key=lambda scene: hashlib.sha256(
                             ("stopreadiness-v1:" + scene).encode()).hexdigest())
    assert len(scene_order) == 54
    scene_split = {"fit": scene_order[:40],
                   "development": scene_order[40:47],
                   "audit": scene_order[47:]}
    indices = {split: [i for i, row in enumerate(records)
                       if row["scene_id"] in set(scenes)]
               for split, scenes in scene_split.items()}
    assert sorted(sum(indices.values(), [])) == list(range(531))
    fit_ids = indices["fit"]
    mean = data[fit_ids].mean(axis=0)
    std = np.maximum(data[fit_ids].std(axis=0), 1e-6)
    x = torch.from_numpy(((data - mean) / std).astype("float32"))
    labels = torch.tensor([
        row["terminal_distance_m_for_replay_audit_only"] <= 3.5
        for row in records], dtype=torch.float32)
    torch.manual_seed(11)
    weight = torch.nn.Parameter(torch.zeros(x.shape[1]))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=100,
                                  line_search_fn="strong_wolfe")
    fit_tensor = torch.tensor(fit_ids)

    def closure():
        optimizer.zero_grad()
        logits = x[fit_tensor] @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, labels[fit_tensor])
        loss = loss + 0.05 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        scores = torch.sigmoid(x @ weight + bias).numpy()
    far_fit = sorted(scores[[i for i in fit_ids if labels[i] == 0]])
    threshold = float(far_fit[int(0.95 * len(far_fit))])
    raw_visual = v1
    report = {
        "schema": "stop_readiness_onpolicy_calibration_screen_v1",
        "interpretation": "Train-scene exploratory diagnostic after inspecting pilot failure; scene-disjoint head splits do not make the frozen backbones independent. No online or val-unseen claim.",
        "manifest_sha256": digest(args.manifest),
        "sft_features_sha256": digest(args.sft_features),
        "visual_scores_sha256": digest(args.visual_scores),
        "feature_names": FEATURES,
        "label": "terminal simulator distance at most 3.5 m, train only",
        "scene_split": scene_split,
        "fit_only_feature_mean": mean.tolist(),
        "fit_only_feature_std": std.tolist(),
        "fit_only_weights": weight.detach().tolist(),
        "fit_only_bias": float(bias.detach()),
        "fit_only_threshold_for_5pct_far_rate": threshold,
        "calibrated": {split: metrics(records, scores, rows, threshold)
                       for split, rows in indices.items()},
        "visual_terminal_reference": {
            split: {"near_vs_far_auc": metrics(records, raw_visual, rows, 0)["near_vs_far_auc"],
                    "near_failed_vs_far_failed_auc": metrics(records, raw_visual, rows, 0)["near_failed_vs_far_failed_auc"]}
            for split, rows in indices.items()},
    }
    dev = report["calibrated"]["development"]
    audit = report["calibrated"]["audit"]
    report["predeclared_screen"] = {
        "rule": "development and audit near-vs-far AUC >=0.70, failed-stop near-vs-far AUC >=0.65, and far false-positive rate <=0.10 at fit-only threshold",
        "pass": all(
            item["near_vs_far_auc"] is not None and item["near_vs_far_auc"] >= .70 and
            item["near_failed_vs_far_failed_auc"] is not None and
            item["near_failed_vs_far_failed_auc"] >= .65 and
            item["far_failed_positive_rate_at_fit_threshold"] <= .10
            for item in (dev, audit)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"calibrated": report["calibrated"],
                      "predeclared_screen": report["predeclared_screen"]},
                     indent=2))


if __name__ == "__main__":
    main()
