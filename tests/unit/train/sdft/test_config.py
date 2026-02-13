import pytest
from pydantic import ValidationError

from prime_rl.trainer.sdft.config import SDFTLossConfig, SDFTTrainerConfig


def test_valid_default_config():
    """Default config parses without errors."""
    config = SDFTTrainerConfig()
    assert config.loss.alpha == 0.0
    assert config.loss.temperature == 1.0


def test_alpha_bounds():
    """Alpha must be in [0, 1]."""
    SDFTLossConfig(alpha=0.0)
    SDFTLossConfig(alpha=0.5)
    SDFTLossConfig(alpha=1.0)

    with pytest.raises(ValidationError):
        SDFTLossConfig(alpha=-0.1)

    with pytest.raises(ValidationError):
        SDFTLossConfig(alpha=1.1)


def test_temperature_positive():
    """Temperature must be positive."""
    SDFTLossConfig(temperature=0.1)
    SDFTLossConfig(temperature=2.0)

    with pytest.raises(ValidationError):
        SDFTLossConfig(temperature=0.0)

    with pytest.raises(ValidationError):
        SDFTLossConfig(temperature=-1.0)


def test_auto_tokenizer_setup():
    """Tokenizer name auto-set from model name."""
    config = SDFTTrainerConfig(model={"name": "Qwen/Qwen3-0.6B"})
    assert config.tokenizer.name == "Qwen/Qwen3-0.6B"


def test_fused_lm_head_disabled():
    """fused_lm_head_chunk_size is forced to 'disabled'."""
    config = SDFTTrainerConfig()
    assert config.model.fused_lm_head_chunk_size == "disabled"


def test_batch_divisibility():
    """batch_size must be divisible by micro_batch_size."""
    with pytest.raises(ValidationError):
        SDFTTrainerConfig(data={"batch_size": 5, "micro_batch_size": 2})


def test_entropy_quantile_bounds():
    """top_entropy_quantile must be in [0, 1]."""
    SDFTLossConfig(top_entropy_quantile=0.0)
    SDFTLossConfig(top_entropy_quantile=1.0)
    SDFTLossConfig(top_entropy_quantile=0.5)

    with pytest.raises(ValidationError):
        SDFTLossConfig(top_entropy_quantile=1.5)

    with pytest.raises(ValidationError):
        SDFTLossConfig(top_entropy_quantile=-0.1)
