import pytest
import torch

from prime_rl.trainer.sdft.loss import entropy_mask, sdft_kl_loss

pytestmark = [pytest.mark.gpu]


def test_forward_kl_known_distributions():
    """Forward KL between known distributions produces positive loss."""
    student = torch.zeros(2, 10, 100, device="cuda")  # uniform logits
    teacher = torch.randn(2, 10, 100, device="cuda") * 5  # peaked
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss, metrics = sdft_kl_loss(student, teacher, mask, alpha=0.0)
    assert loss.shape == ()
    assert loss.item() > 0
    assert "kl_divergence" in metrics


def test_reverse_kl():
    """Reverse KL (alpha=1.0) is different from forward KL."""
    student = torch.randn(2, 10, 50, device="cuda")
    teacher = torch.randn(2, 10, 50, device="cuda") * 3
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    fwd_loss, _ = sdft_kl_loss(student, teacher, mask, alpha=0.0)
    rev_loss, _ = sdft_kl_loss(student, teacher, mask, alpha=1.0)
    assert not torch.allclose(fwd_loss, rev_loss)


def test_jsd_symmetric():
    """JSD with alpha=0.5 is symmetric."""
    a = torch.randn(2, 10, 50, device="cuda")
    b = torch.randn(2, 10, 50, device="cuda")
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss_ab, _ = sdft_kl_loss(a, b, mask, alpha=0.5)
    loss_ba, _ = sdft_kl_loss(b, a, mask, alpha=0.5)
    torch.testing.assert_close(loss_ab, loss_ba, atol=1e-5, rtol=1e-5)


def test_kl_zero_for_identical():
    """KL divergence is 0 when student == teacher."""
    logits = torch.randn(2, 10, 50, device="cuda")
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss, _ = sdft_kl_loss(logits, logits.clone(), mask, alpha=0.0)
    assert loss.item() < 1e-5


def test_completion_mask_applied():
    """Loss only computed on masked (completion) tokens."""
    student = torch.randn(2, 10, 50, device="cuda")
    teacher = torch.randn(2, 10, 50, device="cuda") * 5
    # Only mask last 5 tokens
    mask = torch.zeros(2, 10, dtype=torch.bool, device="cuda")
    mask[:, 5:] = True
    loss_partial, _ = sdft_kl_loss(student, teacher, mask, alpha=0.0)

    # Full mask
    mask_full = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss_full, _ = sdft_kl_loss(student, teacher, mask_full, alpha=0.0)

    # They should differ
    assert not torch.allclose(loss_partial, loss_full)


def test_temperature_scaling():
    """Higher temperature produces softer distributions and lower KL."""
    student = torch.randn(2, 10, 50, device="cuda") * 5
    teacher = torch.randn(2, 10, 50, device="cuda") * 5
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss_t1, _ = sdft_kl_loss(student, teacher, mask, temperature=1.0)
    loss_t5, _ = sdft_kl_loss(student, teacher, mask, temperature=5.0)
    assert loss_t5.item() < loss_t1.item()


def test_gradient_flows_through_student():
    """Gradients flow through student logits, not teacher logits."""
    student = torch.randn(2, 10, 50, device="cuda", requires_grad=True)
    teacher = torch.randn(2, 10, 50, device="cuda", requires_grad=True)
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")

    # Teacher should be detached inside the loss
    loss, _ = sdft_kl_loss(student, teacher.detach(), mask, alpha=0.0)
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().sum() > 0


def test_numerical_stability_extreme_logits():
    """No NaN/Inf with very large or very small logit values."""
    student = torch.randn(2, 10, 50, device="cuda") * 100
    teacher = torch.randn(2, 10, 50, device="cuda") * 100
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    loss, metrics = sdft_kl_loss(student, teacher, mask, alpha=0.0)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_divergence"])


def test_entropy_masking_all():
    """top_quantile=1.0 keeps all tokens."""
    logits = torch.randn(2, 10, 50, device="cuda")
    mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
    result = entropy_mask(logits, mask, top_quantile=1.0)
    assert result.all()


def test_entropy_masking_partial():
    """top_quantile < 1.0 removes some tokens."""
    logits = torch.randn(2, 20, 50, device="cuda")
    mask = torch.ones(2, 20, dtype=torch.bool, device="cuda")
    result = entropy_mask(logits, mask, top_quantile=0.5)
    # Should keep roughly half the tokens
    assert result.sum() < mask.sum()
    assert result.sum() > 0
