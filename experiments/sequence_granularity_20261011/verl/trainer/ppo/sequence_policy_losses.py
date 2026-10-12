"""Sequence-level policy-loss experiments for long VLN action sequences."""
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


def _masked_seq_mean(x, mask):
    w = mask.to(dtype=x.dtype)
    return (x * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)


def _stats(log_ratio, mask, clipfrac):
    kl = verl_F.masked_mean(-log_ratio, mask)
    return clipfrac.detach(), kl.detach(), torch.zeros((), device=log_ratio.device)


def _seq_loss(log_prob, old_log_prob, advantages, mask, eps_low, eps_high):
    token_lr = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    seq_ratio = _masked_seq_mean(token_lr, mask).exp()
    adv = _masked_seq_mean(advantages.detach(), mask)
    clipped = seq_ratio.clamp(1.0 - eps_low, 1.0 + eps_high)
    obj, clip_obj = seq_ratio * adv, clipped * adv
    loss = -torch.minimum(obj, clip_obj).mean()
    active = mask.sum(dim=-1) > 0
    cf = ((obj > clip_obj) & active).float().sum() / active.float().sum().clamp_min(1.0)
    return loss, token_lr, cf


@register_policy_loss("gspo_sequence")
def gspo_sequence(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None):
    loss, lr, cf = _seq_loss(log_prob, old_log_prob, advantages, response_mask, 3e-4, 4e-4)
    return loss, *_stats(lr, response_mask, cf)


@register_policy_loss("gmpo_geomean")
def gmpo_geomean(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None):
    # GMPO: sign-aware clipping in log-ratio space, then geometric pooling.
    clip = float(config.clip_ratio)
    low = float(config.clip_ratio_low if config.clip_ratio_low is not None else clip)
    high = float(config.clip_ratio_high if config.clip_ratio_high is not None else clip)
    lr = log_prob - old_log_prob
    sign = torch.sign(advantages.detach())
    clipped_lr = lr.clamp(-low, high)
    signed_min = torch.minimum(sign * lr, sign * clipped_lr) * sign
    count = response_mask.sum(dim=-1).clamp_min(1.0)
    ratio = torch.exp((signed_min * response_mask).sum(dim=-1) / count)
    advantage = (advantages.detach() * response_mask).sum(dim=-1) / count
    loss = -(advantage * ratio).mean()
    cf = verl_F.masked_mean((lr != clipped_lr).float(), response_mask)
    return loss, *_stats(lr, response_mask, cf)


@register_policy_loss("dhpo_average")
def dhpo_average(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None):
    # Fixed 50/50 token-level and sequence-level clipped policy objectives.
    low, high = float(config.clip_ratio_low), float(config.clip_ratio_high)
    lr = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = lr.exp()
    token_adv = advantages.detach()
    token_obj = ratio * token_adv
    token_clip_obj = ratio.clamp(1.0 - low, 1.0 + high) * token_adv
    token_loss = agg_loss(torch.maximum(-token_obj, -token_clip_obj), response_mask, loss_agg_mode)
    seq_loss, _, _ = _seq_loss(log_prob, old_log_prob, advantages, response_mask, low, high)
    loss = 0.5 * token_loss + 0.5 * seq_loss
    cf = verl_F.masked_mean((token_obj > token_clip_obj).float(), response_mask)
    return loss, *_stats(lr, response_mask, cf)
