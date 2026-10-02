"""Fit failure-aware progress while retaining success and instruction ranking.

All supervision comes from preselected train-scene trajectory pairs.  Only
fit splits update weights; development splits select a checkpoint.  Neither
the failure-rank nor the older success/grounding audit is read here.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from train_temporal_progress_encoder import TemporalPotential, digest, predict


def margins(model: TemporalPotential, pairs: torch.Tensor,
            wrong: torch.Tensor | None, device: torch.device) -> dict[str, torch.Tensor]:
    direct = predict(model, pairs, device)
    result = {"endpoint": direct[:, 0, -1] - direct[:, 1, -1]}
    if wrong is not None:
        switched = predict(model, wrong.unsqueeze(1), device)[:, 0, -1]
        result["grounding"] = direct[:, 0, -1] - switched
    return result


def rates(scores: dict[str, torch.Tensor], indices: list[int]) -> dict:
    selected = torch.tensor(indices, dtype=torch.long)
    return {task: {"hits": int((value[selected] > 0).sum()),
                   "pairs": len(indices),
                   "rate": float((value[selected] > 0).float().mean()),
                   "mean_margin": float(value[selected].mean())}
            for task, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-manifest", type=Path, required=True)
    parser.add_argument("--failure-features", type=Path, required=True)
    parser.add_argument("--old-pair-manifest", type=Path, required=True)
    parser.add_argument("--old-v2-manifest", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--old-swaps", type=Path, required=True)
    parser.add_argument("--old-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30:
        raise ValueError("invalid epoch budget")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    failure_manifest = json.loads(args.failure_manifest.read_text())
    old_pairs = json.loads(args.old_pair_manifest.read_text())["pairs"]
    old_v2 = json.loads(args.old_v2_manifest.read_text())
    if failure_manifest["schema"] != "failure_rank_train_scene_v1" or \
            old_v2["source_manifest_sha256"] != digest(args.old_pair_manifest):
        raise ValueError("pair provenance mismatch")
    failure_pairs = failure_manifest["pairs"]
    f_cache = torch.load(args.failure_features, map_location="cpu", weights_only=False)
    s_cache = torch.load(args.old_features, map_location="cpu", weights_only=False)
    if f_cache["manifest_sha256"] != digest(args.failure_manifest) or \
            s_cache["manifest_sha256"] != digest(args.old_pair_manifest) or \
            f_cache["hidden"].shape != (2 * len(failure_pairs), 4, 2048) or \
            s_cache["hidden"].shape != (2 * len(old_pairs), 4, 2048):
        raise ValueError("feature cache provenance mismatch")
    f_hidden = F.normalize(f_cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    s_hidden = F.normalize(s_cache["hidden"].float(), dim=-1).reshape(-1, 2, 4, 2048)
    old_split = {row["pair_id"]: row["split"] for row in old_v2["pairs"]}
    split_f = {name: [i for i, pair in enumerate(failure_pairs)
                      if pair["split"] == name]
               for name in ("fit", "development", "audit")}
    split_s = {name: [i for i, pair in enumerate(old_pairs)
                      if old_split[pair["pair_id"]] == name]
               for name in ("fit", "development", "audit")}
    if {k: len(v) for k, v in split_f.items()} != {
            "fit": 697, "development": 125, "audit": 106} or \
            {k: len(v) for k, v in split_s.items()} != {
            "fit": 300, "development": 52, "audit": 48}:
        raise ValueError("split count mismatch")
    # Read only fit/development wrong-instruction features.  In particular,
    # the older 48-pair audit is left unopened during checkpoint selection.
    wrong = torch.zeros(len(old_pairs), 4, 2048)
    for name in ("fit", "development"):
        for index in split_s[name]:
            pair_id = old_pairs[index]["pair_id"]
            item = torch.load(args.old_swaps / "records" / f"{pair_id}.pt",
                              map_location="cpu", weights_only=False)
            if item["pair_id"] != pair_id or item["hidden"].shape != (4, 2048):
                raise ValueError(f"wrong-instruction feature mismatch: {pair_id}")
            wrong[index] = F.normalize(item["hidden"].float(), dim=-1)
    checkpoint = torch.load(args.old_encoder, map_location="cpu", weights_only=False)
    if checkpoint["source_manifest_sha256"] != digest(args.old_pair_manifest) or \
            checkpoint["v2_manifest_sha256"] != digest(args.old_v2_manifest):
        raise ValueError("old encoder provenance mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher = TemporalPotential().to(device).eval()
    teacher.load_state_dict(checkpoint["model"])
    model = TemporalPotential().to(device)
    model.load_state_dict(checkpoint["model"])
    teacher_failure = predict(teacher, f_hidden[split_f["fit"]], device)
    teacher_success = predict(teacher, s_hidden[split_s["fit"]], device)
    teacher_wrong = predict(teacher, wrong[split_s["fit"]].unsqueeze(1), device)[:, 0]
    base_failure = margins(teacher, f_hidden[split_f["development"]], None, device)
    base_success = margins(teacher, s_hidden[split_s["development"]],
                           wrong[split_s["development"]], device)
    baseline = {"failure_development": rates(
                    base_failure, list(range(len(split_f["development"])))),
                "success_development": rates(
                    base_success, list(range(len(split_s["development"]))))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.02)
    fit_f = torch.tensor(split_f["fit"], dtype=torch.long)
    fit_s = torch.tensor(split_s["fit"], dtype=torch.long)
    teacher_f_by_index = {int(i): teacher_failure[j] for j, i in enumerate(fit_f)}
    teacher_s_by_index = {int(i): teacher_success[j] for j, i in enumerate(fit_s)}
    teacher_w_by_index = {int(i): teacher_wrong[j] for j, i in enumerate(fit_s)}
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled_f = fit_f[torch.randperm(len(fit_f))]
        shuffled_s = fit_s[torch.randperm(len(fit_s))]
        losses = []
        for batch_index, f_index in enumerate(shuffled_f.split(32)):
            s_index = shuffled_s[(batch_index * 32) % len(shuffled_s):
                                 (batch_index * 32) % len(shuffled_s) + 32]
            if len(s_index) < 16:
                s_index = shuffled_s[:32]
            f_out = model(f_hidden[f_index].reshape(-1, 4, 2048).to(device)) \
                .reshape(-1, 2, 4)
            s_out = model(s_hidden[s_index].reshape(-1, 4, 2048).to(device)) \
                .reshape(-1, 2, 4)
            w_out = model(wrong[s_index].to(device))
            f_margin = f_out[:, 0, -1] - f_out[:, 1, -1]
            s_margin = s_out[:, 0, -1] - s_out[:, 1, -1]
            g_margin = s_out[:, 0, -1] - w_out[:, -1]
            gaps = torch.tensor([failure_pairs[int(i)]["distance_gap_m_for_selection_only"]
                                 for i in f_index], device=device)
            failure_loss = F.softplus((gaps / 5).clamp(.3, 1.5) - f_margin).mean()
            success_loss = F.softplus(.3 - s_margin).mean()
            grounding_loss = F.softplus(.3 - g_margin).mean()
            preserve = (F.smooth_l1_loss(f_out, torch.stack(
                [teacher_f_by_index[int(i)] for i in f_index]).to(device), beta=.2) +
                F.smooth_l1_loss(s_out, torch.stack(
                    [teacher_s_by_index[int(i)] for i in s_index]).to(device), beta=.2) +
                F.smooth_l1_loss(w_out, torch.stack(
                    [teacher_w_by_index[int(i)] for i in s_index]).to(device), beta=.2))
            loss = failure_loss + success_loss + grounding_loss + .15 * preserve
            if not torch.isfinite(loss):
                raise ValueError("nonfinite multi-task ranking loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        candidate_failure = margins(model, f_hidden[split_f["development"]], None, device)
        candidate_success = margins(model, s_hidden[split_s["development"]],
                                    wrong[split_s["development"]], device)
        dev_f = rates(candidate_failure, list(range(len(split_f["development"]))))
        dev_s = rates(candidate_success, list(range(len(split_s["development"]))))
        gate = {"failure_gain_at_least_5pp":
                    dev_f["endpoint"]["rate"] >=
                    baseline["failure_development"]["endpoint"]["rate"] + .05 - 1e-8,
                "success_drop_at_most_2pp":
                    dev_s["endpoint"]["rate"] >=
                    baseline["success_development"]["endpoint"]["rate"] - .02 - 1e-8,
                "grounding_drop_at_most_2pp":
                    dev_s["grounding"]["rate"] >=
                    baseline["success_development"]["grounding"]["rate"] - .02 - 1e-8}
        row = {"epoch": epoch, "loss": sum(losses) / len(losses),
               "failure_development": dev_f, "success_development": dev_s,
               "selection_gate": gate}
        history.append(row)
        print(json.dumps(row), flush=True)
        if all(gate.values()):
            key = (dev_f["endpoint"]["rate"],
                   dev_s["endpoint"]["rate"],
                   dev_s["grounding"]["rate"], -epoch)
            if best is None or key > best[0]:
                best = (key, epoch, {name: value.cpu().clone()
                                     for name, value in model.state_dict().items()})
    report = {"schema": "multitask_failure_success_grounding_development_v1",
              "interpretation": "Fit/development train scenes only; older success-development scenes overlap failure-rank fit scenes. Audits unopened; no online RL or val-unseen result.",
              "failure_manifest_sha256": digest(args.failure_manifest),
              "old_pair_manifest_sha256": digest(args.old_pair_manifest),
              "old_v2_manifest_sha256": digest(args.old_v2_manifest),
              "old_encoder_sha256": digest(args.old_encoder),
              "epochs": args.epochs, "seed": args.seed,
              "baseline_development": baseline,
              "history": history, "selected_epoch": best[1] if best else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if best is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best[2], "selected_epoch": best[1],
                    "failure_manifest_sha256": digest(args.failure_manifest),
                    "old_pair_manifest_sha256": digest(args.old_pair_manifest),
                    "old_v2_manifest_sha256": digest(args.old_v2_manifest),
                    "old_encoder_sha256": digest(args.old_encoder)}, args.checkpoint)
    print(json.dumps({"selected_epoch": report["selected_epoch"],
                      "baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()
