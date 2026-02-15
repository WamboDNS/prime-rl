import torch
import torch.nn.functional as F
from torch import Tensor


def add_tail(log_probs: Tensor) -> Tensor:
    """Add tail probability bucket for top-K distillation.

    Computes log(1 - sum(exp(log_probs))) and appends it as an extra dimension,
    ensuring the distribution sums to 1 over K+1 buckets.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def renorm_topk_log_probs(logp: Tensor) -> Tensor:
    """Renormalize top-K log probs to sum to 1."""
    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
    return logp - logZ


def sdft_kl_loss(
    student_log_probs: Tensor,
    teacher_log_probs: Tensor,
    completion_mask: Tensor,
    alpha: float = 0.5,
    is_ratio: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """KL divergence loss between student and teacher distributions.

    Matches SDPO's compute_self_distillation_loss.

    Args:
        student_log_probs: [batch, seq, K] log_softmax output (top-K+tail or full vocab).
        teacher_log_probs: [batch, seq, K] log_softmax output (same token indices as student).
        completion_mask: [batch, seq] boolean mask for completion tokens.
        alpha: 0=forward KL(teacher||student), 0.5=JSD, 1=reverse KL(student||teacher).
        is_ratio: [batch, seq] per-token importance sampling ratio, already clamped.
    """
    if alpha == 0.0:
        kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        kl_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        alpha_t = torch.tensor(alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log(1 - alpha_t), teacher_log_probs + torch.log(alpha_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        kl_loss = torch.lerp(kl_student, kl_teacher, alpha)

    per_token_loss = kl_loss.sum(dim=-1)

    if is_ratio is not None:
        per_token_loss = per_token_loss * is_ratio

    # Token-mean aggregation (matches SDPO)
    num_tokens = completion_mask.sum().clamp(min=1)
    loss = (per_token_loss * completion_mask).sum() / num_tokens

    metrics = {
        "kl_divergence": loss.detach(),
        "num_completion_tokens": completion_mask.sum().detach().float(),
    }

    return loss, metrics
