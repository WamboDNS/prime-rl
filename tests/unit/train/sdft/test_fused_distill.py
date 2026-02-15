"""Tests for fused chunked top-K distillation."""

import torch
import pytest
from prime_rl.trainer.sdft.fused_distill import (
    chunked_teacher_topk,
    _ChunkedStudentGatherFn,
    fused_distill_topk,
)


def _naive_topk_log_probs(hidden, weight, K):
    """Reference: full materialization of [N, V] logits."""
    logits = (hidden @ weight.t()).float()
    log_probs = logits.log_softmax(dim=-1)
    topk_lp, topk_idx = log_probs.topk(K, dim=-1)
    return topk_idx, topk_lp


def _naive_gather_log_probs(hidden, weight, indices):
    """Reference: gather log probs at given indices from full [N, V]."""
    logits = (hidden @ weight.t()).float()
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, indices)


def test_chunked_teacher_topk_matches_naive():
    torch.manual_seed(42)
    N, H, V, K = 16, 32, 128, 10
    hidden = torch.randn(N, H)
    weight = torch.randn(V, H)

    # Naive
    naive_idx, naive_lp = _naive_topk_log_probs(hidden, weight, K)

    # Chunked with small chunk_size to exercise multiple iterations
    chunked_idx, chunked_lp = chunked_teacher_topk(hidden, weight, K, chunk_size=17)

    # The top-K values should match (indices may differ for tied values, so compare sorted values)
    naive_sorted, _ = naive_lp.sort(dim=-1, descending=True)
    chunked_sorted, _ = chunked_lp.sort(dim=-1, descending=True)
    torch.testing.assert_close(chunked_sorted, naive_sorted, atol=1e-5, rtol=1e-5)

    # Also check that the indices point to the same log probs
    logits = (hidden @ weight.t()).float()
    full_lp = logits.log_softmax(dim=-1)
    recovered_lp = full_lp.gather(-1, chunked_idx)
    torch.testing.assert_close(recovered_lp, chunked_lp, atol=1e-5, rtol=1e-5)


def test_chunked_teacher_topk_chunk_size_larger_than_vocab():
    torch.manual_seed(123)
    N, H, V, K = 8, 16, 50, 5
    hidden = torch.randn(N, H)
    weight = torch.randn(V, H)

    naive_idx, naive_lp = _naive_topk_log_probs(hidden, weight, K)
    chunked_idx, chunked_lp = chunked_teacher_topk(hidden, weight, K, chunk_size=1000)

    naive_sorted, _ = naive_lp.sort(dim=-1, descending=True)
    chunked_sorted, _ = chunked_lp.sort(dim=-1, descending=True)
    torch.testing.assert_close(chunked_sorted, naive_sorted, atol=1e-5, rtol=1e-5)


def test_student_gather_matches_naive():
    torch.manual_seed(42)
    N, H, V, K = 16, 32, 128, 10
    hidden = torch.randn(N, H)
    weight = torch.randn(V, H)

    # Get teacher top-K indices
    topk_idx, _ = chunked_teacher_topk(hidden, weight, K, chunk_size=17)

    # Naive
    naive_lp = _naive_gather_log_probs(hidden, weight, topk_idx)

    # Chunked
    chunked_lp = _ChunkedStudentGatherFn.apply(hidden, weight, topk_idx, 17)

    torch.testing.assert_close(chunked_lp, naive_lp, atol=1e-5, rtol=1e-5)


def test_student_gather_gradient():
    """Verify backward produces correct gradients by comparing against naive autograd."""
    torch.manual_seed(42)
    N, H, V, K = 4, 8, 32, 5

    hidden_data = torch.randn(N, H)
    weight_data = torch.randn(V, H)
    topk_idx = torch.randint(0, V, (N, K))

    # Fused path
    h_fused = hidden_data.clone().requires_grad_(True)
    w_fused = weight_data.clone().requires_grad_(True)
    fused_lp = _ChunkedStudentGatherFn.apply(h_fused, w_fused, topk_idx, 7)
    fused_lp.sum().backward()

    # Naive path
    h_naive = hidden_data.clone().requires_grad_(True)
    w_naive = weight_data.clone().requires_grad_(True)
    naive_lp = _naive_gather_log_probs(h_naive, w_naive, topk_idx)
    naive_lp.sum().backward()

    torch.testing.assert_close(h_fused.grad, h_naive.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w_fused.grad, w_naive.grad, atol=1e-4, rtol=1e-4)


def test_fused_distill_topk_end_to_end():
    torch.manual_seed(42)
    N, H, V, K = 16, 32, 128, 10

    student_hidden = torch.randn(N, H, requires_grad=True)
    teacher_hidden = torch.randn(N, H)
    weight = torch.randn(V, H)

    student_topk_lp, teacher_topk_lp, topk_idx = fused_distill_topk(
        student_hidden, teacher_hidden, weight, K=K, chunk_size=17,
    )

    assert student_topk_lp.shape == (N, K)
    assert teacher_topk_lp.shape == (N, K)
    assert topk_idx.shape == (N, K)

    # Verify teacher values match naive
    naive_teacher_idx, naive_teacher_lp = _naive_topk_log_probs(teacher_hidden, weight, K)
    naive_sorted, _ = naive_teacher_lp.sort(dim=-1, descending=True)
    fused_sorted, _ = teacher_topk_lp.sort(dim=-1, descending=True)
    torch.testing.assert_close(fused_sorted, naive_sorted, atol=1e-5, rtol=1e-5)

    # Verify student values match naive gather at teacher's indices
    naive_student_lp = _naive_gather_log_probs(student_hidden, weight, topk_idx)
    torch.testing.assert_close(student_topk_lp, naive_student_lp, atol=1e-5, rtol=1e-5)

    # Verify gradients flow through student
    loss = student_topk_lp.sum()
    loss.backward()
    assert student_hidden.grad is not None
    assert student_hidden.grad.abs().sum() > 0


def test_fused_distill_student_gradient_matches_naive():
    """Verify that gradients through the fused path match naive full-materialization."""
    torch.manual_seed(42)
    N, H, V, K = 8, 16, 64, 5

    # Two copies of the same hidden states
    hidden_data = torch.randn(N, H)
    teacher_hidden = torch.randn(N, H)
    weight_data = torch.randn(V, H)

    # Get shared teacher indices
    topk_idx, _ = chunked_teacher_topk(teacher_hidden, weight_data, K, chunk_size=11)

    # Fused path
    s_fused = hidden_data.clone().requires_grad_(True)
    w_fused = weight_data.clone().requires_grad_(True)
    fused_lp = _ChunkedStudentGatherFn.apply(s_fused, w_fused, topk_idx, 11)
    fused_lp.sum().backward()

    # Naive path
    s_naive = hidden_data.clone().requires_grad_(True)
    w_naive = weight_data.clone().requires_grad_(True)
    naive_lp = _naive_gather_log_probs(s_naive, w_naive, topk_idx)
    naive_lp.sum().backward()

    torch.testing.assert_close(s_fused.grad, s_naive.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w_fused.grad, w_naive.grad, atol=1e-4, rtol=1e-4)
