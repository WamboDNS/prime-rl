"""Fused chunked top-K KL distillation that never materializes [N, V] logit tensors.

Operates on hidden states [N, H] and lm_head weight [V, H], computing top-K log
probabilities in a memory-efficient chunked fashion. This avoids OOM on large
vocabularies (e.g. Qwen's 152064) by only keeping [N, K] buffers across chunks.
"""

import torch
from torch import Tensor


def _online_logsumexp(m: Tensor, s: Tensor, chunk_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Online logsumexp accumulator (no entropy/weighted-sum needed here).

    Maintains:
      m: running max over all chunks so far  [N]
      s: running sum(exp(x - m))             [N]
    """
    chunk_m = torch.amax(chunk_logits, dim=-1)  # [N]
    m_new = torch.maximum(m, chunk_m)
    exp_old = torch.exp(m - m_new)
    chunk_exp = torch.exp(chunk_logits - m_new.unsqueeze(-1))  # [N, C]
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    return m_new, s_new


@torch.no_grad()
def chunked_teacher_topk(
    hidden: Tensor,
    weight: Tensor,
    K: int,
    chunk_size: int = 2048,
) -> tuple[Tensor, Tensor]:
    """Compute top-K log probabilities from hidden states without materializing [N, V].

    Args:
        hidden: [N, H] hidden states
        weight: [V, H] lm_head weight
        K: number of top tokens to keep
        chunk_size: vocab chunk size for memory efficiency

    Returns:
        topk_indices: [N, K] token indices
        topk_log_probs: [N, K] log probabilities
    """
    n = hidden.shape[0]
    vocab = weight.shape[0]
    device = hidden.device

    # Running top-K candidates
    topk_vals = torch.full((n, K), float("-inf"), device=device, dtype=torch.float32)
    topk_idxs = torch.zeros((n, K), device=device, dtype=torch.long)

    # Running logsumexp state
    m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
    s = torch.zeros((n,), device=device, dtype=torch.float32)

    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        logits = (hidden @ weight[start:end].t()).float()  # [N, C]

        m, s = _online_logsumexp(m, s, logits)

        # Merge this chunk's top-K with running top-K
        c = end - start
        if c >= K:
            chunk_topk_vals, chunk_topk_idxs = logits.topk(K, dim=-1)
        else:
            chunk_topk_vals = logits
            chunk_topk_idxs = torch.arange(c, device=device).unsqueeze(0).expand(n, -1)

        chunk_topk_idxs = chunk_topk_idxs + start

        combined_vals = torch.cat([topk_vals, chunk_topk_vals], dim=-1)  # [N, K+K'] or [N, K+C]
        combined_idxs = torch.cat([topk_idxs, chunk_topk_idxs], dim=-1)

        _, sel = combined_vals.topk(K, dim=-1)
        topk_vals = combined_vals.gather(-1, sel)
        topk_idxs = combined_idxs.gather(-1, sel)

    logZ = m + torch.log(s)  # [N]
    topk_log_probs = topk_vals - logZ.unsqueeze(-1)  # [N, K]

    return topk_idxs, topk_log_probs


class _ChunkedStudentGatherFn(torch.autograd.Function):
    """Gather student log probs at teacher's top-K positions without materializing [N, V].

    Forward: iterate vocab in chunks, gather logits at positions falling in each chunk,
    track logZ via online logsumexp. Return gathered_logits - logZ.

    Backward: recompute logits per chunk (not saved from forward), compute softmax chunk,
    and accumulate gradients. Same recompute pattern as _ChunkedLogProbEntropyFn.
    """

    @staticmethod
    def forward(ctx, hidden: Tensor, weight: Tensor, topk_indices: Tensor, chunk_size: int) -> Tensor:
        n, h = hidden.shape
        vocab = weight.shape[0]
        K = topk_indices.shape[1]
        device = hidden.device

        gathered = torch.zeros((n, K), device=device, dtype=torch.float32)
        m = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
        s = torch.zeros((n,), device=device, dtype=torch.float32)

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            logits = (hidden @ weight[start:end].t()).float()  # [N, C]

            m, s = _online_logsumexp(m, s, logits)

            # Gather logits at indices falling in this chunk
            in_chunk = (topk_indices >= start) & (topk_indices < end)
            if in_chunk.any():
                local_idx = (topk_indices - start).clamp(min=0, max=end - start - 1)
                chunk_gathered = logits.gather(-1, local_idx)  # [N, K]
                gathered = torch.where(in_chunk, chunk_gathered, gathered)

        logZ = m + torch.log(s)
        result = gathered - logZ.unsqueeze(-1)

        ctx.save_for_backward(hidden, weight, topk_indices, logZ)
        ctx.chunk_size = chunk_size

        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        hidden, weight, topk_indices, logZ = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        n, h = hidden.shape
        vocab = weight.shape[0]

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)

        g = grad_output.float()

        # Sum of gradients for the normalization term: each position contributes -p * sum(grad)
        g_sum = g.sum(dim=-1)  # [N]

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]
            logits = (hidden @ w_chunk.t()).float()  # [N, C]

            p = torch.exp(logits - logZ.unsqueeze(-1))  # [N, C] softmax chunk

            # Gradient from normalization: -sum(grad) * p for all positions
            grad_logits = (-g_sum).unsqueeze(-1) * p  # [N, C]

            # Gradient at gathered positions: +grad at those positions
            in_chunk = (topk_indices >= start) & (topk_indices < end)
            if in_chunk.any():
                local_idx = (topk_indices - start).clamp(min=0, max=end - start - 1)
                grad_at_pos = torch.where(in_chunk, g, torch.zeros_like(g))
                grad_logits.scatter_add_(-1, local_idx, grad_at_pos)

            grad_hidden.add_(grad_logits.to(hidden.dtype) @ w_chunk)
            grad_w_chunk = grad_logits.to(weight.dtype).t() @ hidden
            grad_weight[start:end].add_(grad_w_chunk)

        return grad_hidden, grad_weight, None, None


def fused_distill_topk(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    weight: Tensor,
    K: int,
    chunk_size: int = 2048,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute top-K student and teacher log probs without materializing [N, V].

    Args:
        student_hidden: [N, H] student hidden states
        teacher_hidden: [N, H] teacher hidden states
        weight: [V, H] shared lm_head weight
        K: number of top tokens
        chunk_size: vocab chunk size

    Returns:
        student_topk_lp: [N, K] student log probs at teacher's top-K positions (has grad)
        teacher_topk_lp: [N, K] teacher log probs at top-K positions (no grad)
        topk_indices: [N, K] token indices
    """
    topk_indices, teacher_topk_lp = chunked_teacher_topk(teacher_hidden, weight, K, chunk_size)
    student_topk_lp = _ChunkedStudentGatherFn.apply(student_hidden, weight, topk_indices, chunk_size)
    return student_topk_lp, teacher_topk_lp, topk_indices
