import pytest
from pydantic import ValidationError

from prime_rl.trainer.sdft.config import SDFTLossConfig, SDFTTrainerConfig


def test_valid_default_config():
    """Default config parses without errors."""
    config = SDFTTrainerConfig()
    assert config.loss.alpha == 0.5
    assert config.loss.distillation_topk == 100
    assert config.ref_model.update_rate == 0.05
    assert config.reprompt.success_threshold == 1.0
    assert config.generation.num_completions == 8


def test_alpha_bounds():
    """Alpha must be in [0, 1]."""
    SDFTLossConfig(alpha=0.0)
    SDFTLossConfig(alpha=0.5)
    SDFTLossConfig(alpha=1.0)

    with pytest.raises(ValidationError):
        SDFTLossConfig(alpha=-0.1)

    with pytest.raises(ValidationError):
        SDFTLossConfig(alpha=1.1)


def test_is_clip_positive():
    """is_clip must be positive or None."""
    SDFTLossConfig(is_clip=2.0)
    SDFTLossConfig(is_clip=None)

    with pytest.raises(ValidationError):
        SDFTLossConfig(is_clip=0.0)

    with pytest.raises(ValidationError):
        SDFTLossConfig(is_clip=-1.0)


def test_distillation_topk_positive():
    """distillation_topk must be positive or None."""
    SDFTLossConfig(distillation_topk=100)
    SDFTLossConfig(distillation_topk=None)

    with pytest.raises(ValidationError):
        SDFTLossConfig(distillation_topk=0)


def test_auto_tokenizer_setup():
    """Tokenizer name auto-set from model name."""
    config = SDFTTrainerConfig(model={"name": "Qwen/Qwen3-0.6B"})
    assert config.tokenizer.name == "Qwen/Qwen3-0.6B"


def test_fused_lm_head_disabled():
    """fused_lm_head_chunk_size is forced to 'disabled'."""
    config = SDFTTrainerConfig()
    assert config.model.fused_lm_head_chunk_size == "disabled"


def test_mini_batch_size_defaults_to_batch_size():
    """mini_batch_size defaults to batch_size when not set."""
    config = SDFTTrainerConfig(data={"batch_size": 32, "micro_batch_size": 1})
    assert config.data.mini_batch_size == 32


def test_mini_batch_size_explicit():
    """mini_batch_size can be set smaller than batch_size."""
    config = SDFTTrainerConfig(data={"batch_size": 32, "mini_batch_size": 1, "micro_batch_size": 1})
    assert config.data.mini_batch_size == 1


def test_batch_divisibility():
    """batch_size must be divisible by mini_batch_size."""
    with pytest.raises(ValidationError):
        SDFTTrainerConfig(data={"batch_size": 32, "mini_batch_size": 5, "micro_batch_size": 1})


def test_mini_batch_divisible_by_micro():
    """mini_batch_size must be divisible by micro_batch_size."""
    with pytest.raises(ValidationError):
        SDFTTrainerConfig(data={"batch_size": 32, "mini_batch_size": 3, "micro_batch_size": 2})


def test_batch_divisible_by_num_completions():
    """batch_size must be divisible by num_completions."""
    with pytest.raises(ValidationError):
        SDFTTrainerConfig(
            data={"batch_size": 10, "micro_batch_size": 1},
            generation={"num_completions": 3},
        )
