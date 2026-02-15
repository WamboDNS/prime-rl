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
    """Configures the SDFT KL divergence loss (SDPO-style)."""

    alpha: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="KL direction: 0=forward KL(teacher||student), 0.5=JSD, 1=reverse KL(student||teacher)."),
    ] = 0.5

    full_logit_distillation: Annotated[
        bool,
        Field(description="Use full-logit KL distillation (True) or sampled-token-only (False)."),
    ] = True

    distillation_topk: Annotated[
        int | None,
        Field(ge=1, description="Use top-K logits for distillation. None uses full vocabulary."),
    ] = 100

    distillation_add_tail: Annotated[
        bool,
        Field(description="Add tail probability bucket when using top-K distillation."),
    ] = True

    is_clip: Annotated[
        float | None,
        Field(gt=0.0, description="Clip value for IS ratio (π_current/π_old). None disables IS."),
    ] = 2.0


class SDFTRefModelConfig(BaseConfig):
    """Configures the EMA teacher model for SDFT."""

    enabled: Annotated[
        bool,
        Field(description="Whether to use a separate EMA reference model as teacher."),
    ] = False

    update_rate: Annotated[
        float,
        Field(gt=0.0, le=1.0, description="EMA blend: teacher = rate*student + (1-rate)*teacher."),
    ] = 0.05


class SDFTRepromptConfig(BaseConfig):
    """Configures dynamic teacher prompt construction (SDPO-style self-distillation)."""

    success_threshold: Annotated[
        float,
        Field(description="Minimum reward score to consider a completion successful."),
    ] = 1.0

    dont_reprompt_on_self_success: Annotated[
        bool,
        Field(description="Exclude a sample's own success from demonstrations (recommended True)."),
    ] = True

    remove_thinking: Annotated[
        bool,
        Field(description="Strip <think>...</think> from demonstrations."),
    ] = True

    max_reprompt_length: Annotated[
        int,
        Field(ge=1, description="Max tokens for teacher prompt (before completion)."),
    ] = 10240

    reprompt_template: Annotated[
        str,
        Field(description="Template for teacher prompt. Variables: {prompt}, {solution}, {feedback}."),
    ] = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"

    solution_template: Annotated[
        str,
        Field(description="Template for solution section. Variable: {successful_previous_attempt}."),
    ] = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"

    feedback_template: Annotated[
        str,
        Field(description="Template for feedback section. Variable: {feedback_raw}."),
    ] = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"

    include_feedback: Annotated[
        bool,
        Field(description="Include environment feedback in teacher prompt."),
    ] = False


class SDFTGenerationConfig(BaseConfig):
    """Configures completion generation for SDFT."""

    num_completions: Annotated[
        int,
        Field(ge=1, description="Number of completions per prompt (SDPO rollout.n)."),
    ] = 8

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
        Field(description="HuggingFace dataset name or local path (JSON/Parquet/disk)."),
    ] = "PrimeIntellect/Reverse-Text-SFT"

    dataset_split: Annotated[
        str,
        Field(description="Dataset split to use."),
    ] = "train"

    prompt_field: Annotated[
        str,
        Field(description="Field name for the prompt."),
    ] = "prompt"

    answer_field: Annotated[
        str,
        Field(description="Field name for the ground truth answer (used for scoring)."),
    ] = "answer"

    system_field: Annotated[
        str | None,
        Field(description="Field name for optional system message. None to disable."),
    ] = "system"

    kind_field: Annotated[
        str | None,
        Field(description="Field name for task type (mcq, tooluse, code). Used by scoring. None to auto-detect."),
    ] = "kind"

    shuffle: Annotated[
        bool,
        Field(description="Whether to shuffle the dataset at the beginning of each epoch."),
    ] = True

    seed: Annotated[int, Field(description="Random seed for shuffling.")] = 0

    batch_size: Annotated[int, Field(ge=1, description="Total training samples per batch (prompts × num_completions).")] = 32

    mini_batch_size: Annotated[
        int | None,
        Field(ge=1, description="Samples per optimizer step. None defaults to batch_size. Set to 1 for per-sample updates."),
    ] = None

    micro_batch_size: Annotated[int, Field(ge=1, description="Micro batch size for forward/backward.")] = 1


FakeSDFTDataConfig: TypeAlias = SDFTDataConfig


class SDFTTrainerConfig(BaseSettings):
    """Configures the SDFT trainer."""

    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    loss: SDFTLossConfig = SDFTLossConfig()
    ref_model: SDFTRefModelConfig = SDFTRefModelConfig()
    reprompt: SDFTRepromptConfig = SDFTRepromptConfig()
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
    def validate_batch_sizes(self):
        if self.data.batch_size % self.generation.num_completions != 0:
            raise ValueError("batch_size must be divisible by num_completions")
        if self.data.mini_batch_size is None:
            self.data.mini_batch_size = self.data.batch_size
        if self.data.batch_size % self.data.mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by mini_batch_size")
        if self.data.mini_batch_size % self.data.micro_batch_size != 0:
            raise ValueError("mini_batch_size must be divisible by micro_batch_size")
        return self
