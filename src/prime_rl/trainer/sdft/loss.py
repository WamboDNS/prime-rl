from beartype import beartype as typechecker
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=typechecker)
def sdft_kl_loss(
    student_logits: Float[Tensor, "batch seq vocab"],
    teacher_logits: Float[Tensor, "batch seq vocab"],
    completion_mask: Bool[Tensor, "batch seq"],
    alpha: float = 0.0,
    temperature: float = 1.0,
    importance_weights: Float[Tensor, "batch seq"] | None = None,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    """Full-vocabulary KL divergence between student and teacher distributions.

    Args:
        student_logits: Raw logits from the student forward pass.
        teacher_logits: Raw logits from the teacher forward pass (no grad).
        completion_mask: Boolean mask indicating completion tokens to include in loss.
        alpha: KL direction. 0=forward KL(student||teacher), 1=reverse KL(teacher||student),
               (0,1)=Jensen-Shannon divergence.
        temperature: Softmax temperature for smoothing distributions.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    student_logprobs = (student_logits / temperature).log_softmax(dim=-1)
    teacher_logprobs = (teacher_logits / temperature).log_softmax(dim=-1)

    if alpha == 0.0:
        # Forward KL: KL(student || teacher) = sum_x student(x) * (log student(x) - log teacher(x))
        per_token_kl = _forward_kl(student_logprobs, teacher_logprobs)
    elif alpha == 1.0:
        # Reverse KL: KL(teacher || student) = sum_x teacher(x) * (log teacher(x) - log student(x))
        per_token_kl = _forward_kl(teacher_logprobs, student_logprobs)
    else:
        # Jensen-Shannon Divergence
        student_probs = student_logprobs.exp()
        teacher_probs = teacher_logprobs.exp()
        mix_probs = (1 - alpha) * student_probs + alpha * teacher_probs
        mix_logprobs = mix_probs.log()
        kl_mix_teacher = (teacher_probs * (teacher_logprobs - mix_logprobs)).sum(dim=-1)
        kl_mix_student = (student_probs * (student_logprobs - mix_logprobs)).sum(dim=-1)
        per_token_kl = alpha * kl_mix_teacher + (1 - alpha) * kl_mix_student

    if importance_weights is not None:
        per_token_kl = per_token_kl * importance_weights

    # Mean over completion tokens, then mean over batch
    num_tokens = completion_mask.sum(dim=-1).clamp(min=1)
    per_sample_loss = (per_token_kl * completion_mask).sum(dim=-1) / num_tokens
    loss = per_sample_loss.mean()

    metrics = {
        "kl_divergence": loss.detach(),
        "num_completion_tokens": completion_mask.sum().detach().float(),
        "per_token_kl_mean": (per_token_kl * completion_mask).sum().detach() / completion_mask.sum().clamp(min=1),
    }

    return loss, metrics


def _forward_kl(
    p_logprobs: Float[Tensor, "batch seq vocab"],
    q_logprobs: Float[Tensor, "batch seq vocab"],
) -> Float[Tensor, "batch seq"]:
    """KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x)), summed over vocab."""
    return (p_logprobs.exp() * (p_logprobs - q_logprobs)).sum(dim=-1)


@jaxtyped(typechecker=typechecker)
def entropy_mask(
    logits: Float[Tensor, "batch seq vocab"],
    completion_mask: Bool[Tensor, "batch seq"],
    top_quantile: float,
) -> Bool[Tensor, "batch seq"]:
    """Mask out low-entropy tokens, keeping only the top-quantile highest entropy tokens.

    Args:
        logits: Logits to compute entropy over.
        completion_mask: Existing completion mask.
        top_quantile: Fraction of tokens to keep (e.g., 0.8 = top 80% by entropy).

    Returns:
        Updated mask that zeros out low-entropy positions.
    """
    if top_quantile >= 1.0:
        return completion_mask

    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)

    # Set entropy of non-completion tokens to -inf so they don't affect quantile
    token_entropy = token_entropy.masked_fill(~completion_mask, float("-inf"))

    # Compute per-sample quantile threshold
    num_completion = completion_mask.sum(dim=-1)
    k = (num_completion.float() * (1 - top_quantile)).long().clamp(min=1)

    # For each sample, find the k-th smallest entropy among completion tokens
    sorted_entropy, _ = token_entropy.sort(dim=-1)
    threshold = sorted_entropy.gather(dim=-1, index=k.unsqueeze(-1)).squeeze(-1)

    entropy_keep = token_entropy >= threshold.unsqueeze(-1)
    return completion_mask & entropy_keep
