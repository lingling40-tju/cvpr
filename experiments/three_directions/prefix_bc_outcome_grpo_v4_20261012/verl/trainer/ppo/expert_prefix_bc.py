"""Prefix-only imitation mask for exact expert-aligned on-policy turns.

Credits only the contiguous generated action prefix that exactly matches
the frozen R2R expert action sequence after the already-executed history.
Once one turn diverges, later turns receive no imitation label.
"""
from __future__ import annotations

import torch


def expert_aligned_prefix_mask(data, response_mask: torch.Tensor):
    batch = data.batch
    non_tensor = data.non_tensor_batch
    required = ("action_mask", "responses")
    if any(key not in batch for key in required):
        raise ValueError("missing response/action masks")
    if not all(key in non_tensor for key in ("info", "extra_info", "history_actions")):
        raise ValueError("missing trajectory metadata for prefix imitation")
    infos = non_tensor["info"]
    extras = non_tensor["extra_info"]
    histories = non_tensor["history_actions"]
    n, length = response_mask.shape
    if not (len(infos) == len(extras) == len(histories) == n):
        raise ValueError("batch metadata size mismatch")
    action_mask = batch["action_mask"][:, -length:].bool()
    if action_mask.shape != response_mask.shape:
        raise ValueError("action mask shape mismatch")
    out = torch.zeros_like(response_mask, dtype=torch.bool)
    aligned_rows = aligned_turns = total_turns = 0
    for row, (info, extra, history) in enumerate(zip(infos, extras, histories)):
        gt = list(extra.get("gt_actions", []))
        cursor = len(list(history))
        turns = info.get("gen_traj", [])
        positions = torch.nonzero(action_mask[row], as_tuple=False).flatten().tolist()
        spans = []
        for pos in positions:
            if not spans or pos != spans[-1][-1] + 1:
                spans.append([pos])
            else:
                spans[-1].append(pos)
        if len(spans) != len(turns):
            raise ValueError(f"turn/action span mismatch at row {row}: {len(spans)} vs {len(turns)}")
        row_turns = 0
        for turn, span in zip(turns, spans):
            total_turns += 1
            actions = list(turn.get("executed_actions", []))
            if not actions or any(a == "stop" for a in actions):
                break
            if gt[cursor:cursor + len(actions)] != actions:
                break
            token_idx = torch.tensor(span, dtype=torch.long, device=out.device)
            out[row, token_idx] = response_mask[row, token_idx].bool()
            cursor += len(actions)
            row_turns += 1
            aligned_turns += 1
        aligned_rows += int(row_turns > 0)
    metrics = {
        "rows_with_aligned_prefix": aligned_rows,
        "aligned_prefix_turns": aligned_turns,
        "observed_generated_turns": total_turns,
        "imitation_token_fraction": float(out.sum().item()) / max(1, int(response_mask.sum().item())),
    }
    return out, metrics


def synthetic_self_check():
    gt = ["turn left 30 degrees", "move forward 50cm", "turn right 15 degrees"]
    generated = [
        ["turn left 30 degrees"],
        ["move forward 50cm"],
        ["turn left 15 degrees"],
    ]
    cursor = 0
    mask = []
    for actions in generated:
        if gt[cursor:cursor + len(actions)] != actions:
            break
        mask.append(actions)
        cursor += len(actions)
    assert mask == generated[:2]
    assert gt[cursor:cursor + 1] != generated[2]
    return {"exact_prefix_kept": 2, "divergent_turn_and_suffix_excluded": True}
