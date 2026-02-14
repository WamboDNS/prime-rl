from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)
from prime_rl.utils.config import ClientConfig, LogConfig, WandbConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class SDFTLossConfig(BaseConfig):
    """Configures the SDFT KL divergence loss."""

    alpha: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="KL direction: 0=forward KL, 1=reverse KL, (0,1)=JSD."),
    ] = 0.0

    temperature: Annotated[
        float,
        Field(gt=0.0, description="Softmax temperature for smoothing distributions."),
    ] = 1.0

    top_entropy_quantile: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Keep only top-entropy tokens. 1.0=all tokens, 0.8=top 80%."),
    ] = 1.0

    num_loss_tokens_to_skip: Annotated[
        int,
        Field(ge=0, description="Skip first N completion tokens in the loss."),
    ] = 0

    importance_sampling: Annotated[
        bool,
        Field(description="Correct for distribution mismatch between vLLM sampler and training model."),
    ] = True

    importance_sampling_cap: Annotated[
        float,
        Field(gt=0.0, description="Truncation cap for per-token importance ratio."),
    ] = 2.0


class SDFTRefModelConfig(BaseConfig):
    """Configures the optional EMA reference model for SDFT."""

    enabled: Annotated[
        bool,
        Field(description="Whether to use a separate EMA reference model as teacher."),
    ] = False

    sync_steps: Annotated[
        int,
        Field(ge=1, description="Sync EMA reference model every N steps."),
    ] = 1

    mixup_alpha: Annotated[
        float,
        Field(gt=0.0, le=1.0, description="EMA blend: ref = alpha*student + (1-alpha)*ref."),
    ] = 0.01


class SDFTGenerationConfig(BaseConfig):
    """Configures completion generation for SDFT."""

    generate_from_teacher: Annotated[
        bool,
        Field(description="Generate completions from teacher prompt (True) or student prompt (False)."),
    ] = False

    num_iterations: Annotated[
        int,
        Field(ge=1, description="Number of gradient steps per generation batch."),
    ] = 1

    max_completion_length: Annotated[
        int,
        Field(ge=1, description="Maximum number of tokens to generate per completion."),
    ] = 1024

    max_prompt_length: Annotated[
        int,
        Field(ge=1, description="Maximum number of tokens in prompt before truncation."),
    ] = 1024

    temperature: Annotated[
        float,
        Field(gt=0.0, description="Sampling temperature for generation."),
    ] = 1.0

    top_p: Annotated[
        float,
        Field(gt=0.0, le=1.0, description="Top-p (nucleus) sampling parameter."),
    ] = 1.0

    top_k: Annotated[
        int,
        Field(description="Top-k sampling parameter. -1 to disable."),
    ] = -1


class SDFTDataConfig(BaseConfig):
    """Configures the SDFT dataset."""

    type: Literal["sdft", "fake"] = "sdft"

    dataset_name: Annotated[
        str,
        Field(description="HuggingFace dataset name or path."),
    ] = "PrimeIntellect/Reverse-Text-SFT"

    dataset_split: Annotated[
        str,
        Field(description="Dataset split to use."),
    ] = "train"

    prompt_field: Annotated[
        str,
        Field(description="Field name for student prompt."),
    ] = "prompt"

    teacher_prompt_field: Annotated[
        str,
        Field(description="Field name for teacher prompt."),
    ] = "teacher_prompt"

    shuffle: Annotated[
        bool,
        Field(description="Whether to shuffle the dataset at the beginning of each epoch."),
    ] = True

    seed: Annotated[int, Field(description="Random seed for shuffling.")] = 0

    batch_size: Annotated[int, Field(ge=1, description="Number of samples per generation batch.")] = 4

    micro_batch_size: Annotated[int, Field(ge=1, description="Micro batch size for forward/backward.")] = 1


FakeSDFTDataConfig: TypeAlias = SDFTDataConfig


class SDFTTrainerConfig(BaseSettings):
    """Configures the SDFT trainer."""

    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    loss: SDFTLossConfig = SDFTLossConfig()
    ref_model: SDFTRefModelConfig = SDFTRefModelConfig()
    generation: SDFTGenerationConfig = SDFTGenerationConfig()
    data: SDFTDataConfig = SDFTDataConfig()
    client: ClientConfig = ClientConfig()

    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()

    ckpt: CheckpointConfig | None = None
    log: LogConfig = LogConfig()
    wandb: WandbConfig | None = None

    output_dir: Annotated[
        Path,
        Field(description="Directory to write outputs to."),
    ] = Path("outputs")

    max_steps: Annotated[
        int | None,
        Field(description="Maximum number of training steps. If None, runs indefinitely."),
    ] = None

    dist_timeout_seconds: Annotated[
        int,
        Field(description="Timeout in seconds for torch distributed ops."),
    ] = 600

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def validate_and_disable_chunked_loss(self):
        if isinstance(self.model.fused_lm_head_chunk_size, int):
            raise ValueError(
                "Chunked loss is not supported for SDFT training, please set `model.fused_lm_head_chunk_size` to 'disabled'"
            )
        self.model.fused_lm_head_chunk_size = "disabled"
        return self

    @model_validator(mode="after")
    def validate_batch_divisibility(self):
        if self.data.batch_size % self.data.micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        return self
