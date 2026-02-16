import pytest
import torch

from prime_rl.trainer.sdft.loss import add_tail, renorm_topk_log_probs, sdft_kl_loss

pytestmark = [pytest.mark.gpu]


def test_forward_kl_known_distributions():
    """Forward KL between known distributions produces positive loss."""
    student = torch.randn(2, 10, 100, device="cuda").log_softmax(dim=-1)
    teacher = (torch.randn(2, 10, 100, device="cuda") * 5).log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss, metrics = sdft_kl_loss(student, teacher, mask, alpha=0.0)
    assert loss.shape == ()
    assert loss.item() > 0
    assert "kl_divergence" in metrics


def test_reverse_kl():
    """Reverse KL (alpha=1.0) is different from forward KL."""
    student = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    teacher = (torch.randn(2, 10, 50, device="cuda") * 3).log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    fwd_loss, _ = sdft_kl_loss(student, teacher, mask, alpha=0.0)
    rev_loss, _ = sdft_kl_loss(student, teacher, mask, alpha=1.0)
    assert not torch.allclose(fwd_loss, rev_loss)


def test_jsd_symmetric():
    """JSD with alpha=0.5 is symmetric."""
    a = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    b = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss_ab, _ = sdft_kl_loss(a, b, mask, alpha=0.5)
    loss_ba, _ = sdft_kl_loss(b, a, mask, alpha=0.5)
    torch.testing.assert_close(loss_ab, loss_ba, atol=1e-5, rtol=1e-5)


def test_kl_zero_for_identical():
    """KL divergence is 0 when student == teacher."""
    log_probs = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss, _ = sdft_kl_loss(log_probs, log_probs.clone(), mask, alpha=0.0)
    assert loss.item() < 1e-5


def test_completion_mask_applied():
    """Loss only computed on masked (completion) tokens."""
    student = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    teacher = (torch.randn(2, 10, 50, device="cuda") * 5).log_softmax(dim=-1)
    mask = torch.zeros(2, 10, dtype=torch.bool, device="cuda")
    mask[:, 5:] = True
    loss_partial, _ = sdft_kl_loss(student, teacher, mask, alpha=0.0)

    mask_full = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss_full, _ = sdft_kl_loss(student, teacher, mask_full, alpha=0.0)

    assert not torch.allclose(loss_partial, loss_full)


def test_gradient_flows_through_student():
    """Gradients flow through student log_probs."""
    student = torch.randn(2, 10, 50, device="cuda", requires_grad=True)
    student_lp = student.log_softmax(dim=-1)
    teacher_lp = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")

    loss, _ = sdft_kl_loss(student_lp, teacher_lp.detach(), mask, alpha=0.0)
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0


def test_numerical_stability_extreme_logits():
    """No NaN/Inf with very large logit values."""
    student = (torch.randn(2, 10, 50, device="cuda") * 100).log_softmax(dim=-1)
    teacher = (torch.randn(2, 10, 50, device="cuda") * 100).log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss, metrics = sdft_kl_loss(student, teacher, mask, alpha=0.0)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_divergence"])


def test_is_ratio_applied():
    """IS ratio modifies the loss."""
    student = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    teacher = (torch.randn(2, 10, 50, device="cuda") * 3).log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")

    loss_no_is, _ = sdft_kl_loss(student, teacher, mask, alpha=0.5)
    is_ratio = torch.full((2, 10), 2.0, device="cuda")
    loss_with_is, _ = sdft_kl_loss(student, teacher, mask, alpha=0.5, is_ratio=is_ratio)

    torch.testing.assert_close(loss_with_is, loss_no_is * 2.0, atol=1e-5, rtol=1e-5)


def test_rollout_is_weights_applied():
    """Rollout-correction IS weights modify the loss."""
    student = torch.randn(2, 10, 50, device="cuda").log_softmax(dim=-1)
    teacher = (torch.randn(2, 10, 50, device="cuda") * 3).log_softmax(dim=-1)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")

    loss_no_rollout_is, _ = sdft_kl_loss(student, teacher, mask, alpha=0.5)
    rollout_is_weights = torch.full((2, 10), 1.5, device="cuda")
    loss_with_rollout_is, _ = sdft_kl_loss(
        student,
        teacher,
        mask,
        alpha=0.5,
        rollout_is_weights=rollout_is_weights,
    )

    torch.testing.assert_close(loss_with_rollout_is, loss_no_rollout_is * 1.5, atol=1e-5, rtol=1e-5)


def test_add_tail():
    """add_tail appends a valid tail probability bucket."""
    log_probs = torch.tensor([[-1.0, -2.0, -3.0]]).log_softmax(dim=-1)
    with_tail = add_tail(log_probs)
    assert with_tail.shape[-1] == 4
    probs = with_tail.exp().sum(dim=-1)
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-5, rtol=1e-5)


def test_renorm_topk():
    """renorm_topk_log_probs normalizes to sum to 1."""
    log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
    renormed = renorm_topk_log_probs(log_probs)
    probs = renormed.exp().sum(dim=-1)
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-5, rtol=1e-5)
