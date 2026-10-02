"""Fit and evaluate a small train-only instruction-conditioned progress head.

No RL is launched here. The calibration episodes come from ten R2R train
scenes excluded from fitting, and their metrics are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressHead(nn.Module):
    def __init__(self, dim: int, representation: str):
        super().__init__()
        if representation not in ("current_only", "start_relative"):
            raise ValueError(representation)
        self.representation = representation
        input_dim = dim * (4 if representation == "current_only" else 5)
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 128),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, image: torch.Tensor, text: torch.Tensor,
                start: torch.Tensor) -> torch.Tensor:
        if self.representation == "current_only":
            x = torch.cat((image, text, image * text, torch.abs(image - text)), dim=-1)
        else:
            displacement = image - start
            x = torch.cat((image, text, start, displacement,
                           displacement * text), dim=-1)
        return self.net(x).squeeze(-1).sigmoid()


def load_features(path: Path, manifest: dict, subset: str, device: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["subset"] != subset:
        raise ValueError("feature subset mismatch")
    expected = manifest["subsets"][subset]["episode_ids"]
    ids = data["episode_ids"].tolist()
    if not set(ids).issubset(set(expected)) or \
            set(data["frame_episode_ids"].tolist()) != set(ids):
        raise ValueError("feature episode coverage mismatch")
    if not all(torch.isfinite(data[k]).all() for k in ("texts", "images", "progress_fractions")):
        raise ValueError("nonfinite features")
    data = {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()}
    episode_to_text = {episode_id: index for index, episode_id in enumerate(ids)}
    episode_to_frames = defaultdict(list)
    for index, episode_id in enumerate(data["frame_episode_ids"].tolist()):
        episode_to_frames[episode_id].append(index)
    order = []
    for frames in episode_to_frames.values():
        frames.sort(key=lambda i: data["frame_offsets"][i].item())
        order.extend((earlier, later) for earlier in frames for later in frames
                     if data["frame_offsets"][earlier] < data["frame_offsets"][later])
    cf = []
    for pair in manifest["subsets"][subset]["pairs"]:
        a, b = pair["left"], pair["right"]
        if a not in episode_to_frames or b not in episode_to_frames:
            continue
        # Late frames carry goal evidence while initial frames are identical.
        for owner, other in ((a, b), (b, a)):
            for image_index in episode_to_frames[owner][-2:]:
                cf.append((image_index, episode_to_text[owner], episode_to_text[other]))
    if not order or not cf:
        raise ValueError(f"insufficient temporal or counterfactual pairs in {subset}")
    data["frame_text_indices"] = torch.tensor(
        [episode_to_text[i] for i in data["frame_episode_ids"].tolist()], device=device)
    first_frame = {episode_id: frames[0] for episode_id, frames in episode_to_frames.items()}
    data["frame_start_indices"] = torch.tensor(
        [first_frame[i] for i in data["frame_episode_ids"].tolist()], device=device)
    data["order_pairs"] = torch.tensor(order, dtype=torch.long, device=device)
    data["counterfactual_pairs"] = torch.tensor(cf, dtype=torch.long, device=device)
    return data


@torch.no_grad()
def evaluate(model: ProgressHead, data: dict) -> dict:
    model.eval()
    images, texts = data["images"], data["texts"]
    starts = images[data["frame_start_indices"]]
    own = model(images, texts[data["frame_text_indices"]], starts)
    oi, oj = data["order_pairs"].T
    ci, ct, wt = data["counterfactual_pairs"].T
    correct = model(images[ci], texts[ct], starts[ci])
    wrong = model(images[ci], texts[wt], starts[ci])
    raw_own = (images * texts[data["frame_text_indices"]]).sum(-1)
    raw_correct = (images[ci] * texts[ct]).sum(-1)
    raw_wrong = (images[ci] * texts[wt]).sum(-1)
    return {
        "episodes": len(data["episode_ids"]),
        "frames": len(images),
        "ordered_pairs": len(oi),
        "counterfactual_comparisons": len(ci),
        "ordinal_accuracy": (own[oj] > own[oi]).float().mean().item(),
        "counterfactual_accuracy": (correct > wrong).float().mean().item(),
        "progress_mse": F.mse_loss(own, data["progress_fractions"]).item(),
        "counterfactual_margin_mean": (correct - wrong).mean().item(),
        "raw_backbone_ordinal_accuracy": (raw_own[oj] > raw_own[oi]).float().mean().item(),
        "raw_backbone_counterfactual_accuracy": (raw_correct > raw_wrong).float().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-features", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--representation", choices=("current_only", "start_relative"),
                        default="start_relative")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = json.loads(args.manifest.read_text())
    fit = load_features(args.fit_features, manifest, "fit", device)
    calibration = load_features(args.calibration_features, manifest, "calibration", device)
    if fit["images"].shape[1] != calibration["images"].shape[1]:
        raise ValueError("backbone feature dimensions differ")
    if fit["model_config_sha256"] != calibration["model_config_sha256"]:
        raise ValueError("fit and calibration use different visual models")
    if fit["backbone"] != calibration["backbone"]:
        raise ValueError("fit and calibration use different backbones")
    model = ProgressHead(fit["images"].shape[1], args.representation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    best, best_step, best_metrics, best_state = float("-inf"), 0, None, None
    generator = torch.Generator(device=device).manual_seed(args.seed)
    n_frame, n_order, n_cf = (len(fit["images"]), len(fit["order_pairs"]),
                              len(fit["counterfactual_pairs"]))
    for step in range(1, args.steps + 1):
        model.train()
        frame_idx = torch.randint(n_frame, (256,), generator=generator, device=device)
        order_idx = torch.randint(n_order, (256,), generator=generator, device=device)
        cf_idx = torch.randint(n_cf, (128,), generator=generator, device=device)
        oi, oj = fit["order_pairs"][order_idx].T
        ci, ct, wt = fit["counterfactual_pairs"][cf_idx].T
        own = model(fit["images"][frame_idx],
                    fit["texts"][fit["frame_text_indices"][frame_idx]],
                    fit["images"][fit["frame_start_indices"][frame_idx]])
        early = model(fit["images"][oi],
                      fit["texts"][fit["frame_text_indices"][oi]],
                      fit["images"][fit["frame_start_indices"][oi]])
        late = model(fit["images"][oj],
                     fit["texts"][fit["frame_text_indices"][oj]],
                     fit["images"][fit["frame_start_indices"][oj]])
        correct = model(fit["images"][ci], fit["texts"][ct],
                        fit["images"][fit["frame_start_indices"][ci]])
        wrong = model(fit["images"][ci], fit["texts"][wt],
                      fit["images"][fit["frame_start_indices"][ci]])
        loss = (F.mse_loss(own, fit["progress_fractions"][frame_idx])
                + 0.4 * F.softplus(0.08 - (late - early)).mean()
                + 0.4 * F.softplus(0.08 - (correct - wrong)).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            metrics = evaluate(model, calibration)
            score = metrics["ordinal_accuracy"] + metrics["counterfactual_accuracy"] - metrics["progress_mse"]
            if score > best:
                best, best_step, best_metrics = score, step, metrics
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            print(json.dumps({"step": step, "loss": loss.item(), "calibration": metrics}), flush=True)
            if step - best_step >= 150:
                break
    if best_state is None:
        raise RuntimeError("no evaluated checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feature_dim": fit["images"].shape[1],
                "representation": args.representation,
                "model_config_sha256": fit["model_config_sha256"],
                "backbone": fit["backbone"],
                "seed": args.seed, "best_step": best_step}, args.output_dir / "head.pt")
    report = {"interpretation": "Train-only offline representation validation; no navigation result.",
              "seed": args.seed, "best_step": best_step,
              "representation": args.representation,
              "backbone": fit["backbone"],
              "fit_episodes": len(fit["episode_ids"]),
              "calibration_episodes": len(calibration["episode_ids"]),
              "calibration": best_metrics}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
