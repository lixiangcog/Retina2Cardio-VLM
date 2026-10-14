"""Use the installed Apache-2.0 veRL primitives, not report-text GRPO.

Sequential single-device QLoRA adapter: not veRL distributed actor scheduling.
The group mean is removed by veRL, then normalized by the GLOBAL batch reward
standard deviation (sample std, correction=1). The paper does not specify the
std correction; this convention is recorded explicitly for reproducibility.
"""
import numpy as np


def advantages(rewards, groups):
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    r = torch.as_tensor(rewards, dtype=torch.float32)
    idx = np.asarray(groups)
    if r.ndim != 1 or len(idx) != len(r) or not torch.isfinite(r).all():
        raise ValueError("Finite full-batch rewards and aligned group ids required")
    if any(int((idx == g).sum()) != 8 for g in set(idx)):
        raise ValueError("All prompt groups must contain exactly 8 actual rollouts")
    centered, _ = compute_grpo_outcome_advantage(
        r[:, None].clone(), torch.ones_like(r[:, None]), idx, norm_adv_by_std_in_grpo=False)
    return (centered[:, 0]/(r.std(correction=1)+1e-6)).detach()


def clipped_loss(logp, old_logp, ref_logp, advantage, *, beta=.005):
    import torch
    from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty
    if logp.ndim != 1 or len(logp) == 0 or any(x.shape != logp.shape for x in (old_logp, ref_logp)):
        raise ValueError("Aligned response-token log probabilities required")
    if not all(torch.isfinite(x).all() for x in (logp, old_logp, ref_logp)):
        raise ValueError("Nonfinite log probability")
    mask = torch.ones_like(logp[None])
    a = torch.full_like(logp[None], float(advantage))
    # Cap exceeds exp(20), veRL's maximum ratio, so no extra dual-clipping is
    # active. A finite value avoids 0*inf NaNs for zero-advantage responses.
    pg, *_ = compute_policy_loss(old_logp[None].detach(), logp[None], a, mask,
                                cliprange_low=.2, cliprange_high=.28,
                                clip_ratio_c=1e9, loss_agg_mode="seq-mean-token-mean")
    kl = kl_penalty(logp, ref_logp.detach(), kl_penalty="low_var_kl").mean()
    return pg + beta*kl, {"policy": float(pg.detach()), "kl": float(kl.detach())}
