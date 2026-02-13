from pathlib import Path
from typing import TypedDict, cast

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch import Tensor
from torch.utils.data import IterableDataset
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sdft.config import SDFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class SDFTPromptBatch(TypedDict):
    """A batch of dual prompts for SDFT."""

    student_prompts: list[str]
    teacher_prompts: list[str]


class SDFTTrainBatch(TypedDict):
    """A tokenized training batch for SDFT with dual inputs."""

    student_input_ids: Tensor  # [batch, seq]
    student_position_ids: Tensor  # [batch, seq]
    teacher_input_ids: Tensor  # [batch, seq]
    teacher_position_ids: Tensor  # [batch, seq]
    completion_mask: Tensor  # [batch, seq] (aligned with student)
    teacher_completion_mask: Tensor  # [batch, seq] (aligned with teacher)
    labels: Tensor  # [batch, seq] target token ids


class SDFTDataset(IterableDataset):
    """Dataset that yields dual-prompt pairs for SDFT training."""

    def __init__(
        self,
        dataset: Dataset,
        prompt_field: str = "prompt",
        teacher_prompt_field: str = "teacher_prompt",
        shuffle: bool = True,
        seed: int = 0,
        batch_size: int = 4,
    ):
        self.logger = get_logger()
        self.dataset = dataset
        self.prompt_field = prompt_field
        self.teacher_prompt_field = teacher_prompt_field
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.num_examples = len(dataset)
        self.step = 0
        self.epoch = 0

        world = get_world()
        self.data_rank = world.rank
        self.data_world_size = world.world_size

    def __iter__(self):
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
        while True:
            epoch = self.step // self.num_examples
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            batch_student_prompts = []
            batch_teacher_prompts = []

            while len(batch_student_prompts) < self.batch_size:
                idx = self.step % self.num_examples

                # Skip samples not for this rank
                if self.step % self.data_world_size == self.data_rank:
                    example = dataset[idx]
                    student_prompt = example[self.prompt_field]
                    teacher_prompt = example[self.teacher_prompt_field]
                    batch_student_prompts.append(student_prompt)
                    batch_teacher_prompts.append(teacher_prompt)

                self.step += 1
                epoch = self.step // self.num_examples
                if epoch > self.epoch:
                    self.epoch = epoch
                    dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            yield SDFTPromptBatch(
                student_prompts=batch_student_prompts,
                teacher_prompts=batch_teacher_prompts,
            )


class FakeSDFTDataset(IterableDataset):
    """Fake dataset for testing SDFT without real data."""

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.step = 0
        self.epoch = 0
        self.num_examples = 100

    def __iter__(self):
        while True:
            student_prompts = [f"What is {i}+{i}?" for i in range(self.step, self.step + self.batch_size)]
            teacher_prompts = [
                f"What is {i}+{i}?\n\nExample answer: {2*i}\n\nNow answer:"
                for i in range(self.step, self.step + self.batch_size)
            ]
            self.step += self.batch_size
            yield SDFTPromptBatch(
                student_prompts=student_prompts,
                teacher_prompts=teacher_prompts,
            )


def setup_sdft_dataset(config: SDFTDataConfig) -> SDFTDataset | FakeSDFTDataset:
    if config.type == "fake":
        return FakeSDFTDataset(batch_size=config.batch_size)

    logger = get_logger()
    logger.info(f"Loading SDFT dataset: {config.dataset_name} (split={config.dataset_split})")
    if Path(config.dataset_name).is_dir():
        dataset = cast(Dataset, load_from_disk(config.dataset_name))
    else:
        dataset = cast(Dataset, load_dataset(config.dataset_name, split=config.dataset_split))

    assert config.prompt_field in dataset.column_names, (
        f"Dataset must have a '{config.prompt_field}' column, found: {dataset.column_names}"
    )
    assert config.teacher_prompt_field in dataset.column_names, (
        f"Dataset must have a '{config.teacher_prompt_field}' column, found: {dataset.column_names}"
    )

    return SDFTDataset(
        dataset=dataset,
        prompt_field=config.prompt_field,
        teacher_prompt_field=config.teacher_prompt_field,
        shuffle=config.shuffle,
        seed=config.seed,
        batch_size=config.batch_size,
    )


def prepare_sdft_batch(
    student_prompts: list[str],
    teacher_prompts: list[str],
    completions: list[str],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    max_completion_length: int,
    num_loss_tokens_to_skip: int = 0,
) -> SDFTTrainBatch:
    """Tokenize dual prompts + shared completions into padded training tensors.

    Both student and teacher get the same completion appended. The completion_mask
    identifies which tokens are from the completion (where we compute KL loss).

    Returns:
        SDFTTrainBatch with all tensors on CPU.
    """
    batch_size = len(student_prompts)

    all_student_ids = []
    all_teacher_ids = []
    all_completion_lengths = []

    for i in range(batch_size):
        # Apply chat template to prompts (matches how vLLM processes them during generation)
        student_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": student_prompts[i]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        teacher_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_prompts[i]}],
            add_generation_prompt=True,
            tokenize=False,
        )

        student_prompt_ids = tokenizer.encode(student_prompt_text, add_special_tokens=False)
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completions[i], add_special_tokens=False)

        # Truncate
        student_prompt_ids = student_prompt_ids[-max_prompt_length:]
        teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]
        completion_ids = completion_ids[:max_completion_length]

        all_student_ids.append(student_prompt_ids + completion_ids)
        all_teacher_ids.append(teacher_prompt_ids + completion_ids)
        all_completion_lengths.append(len(completion_ids))

    # Pad to max length in batch (left-pad prompts, right-pad after completion)
    max_student_len = max(len(ids) for ids in all_student_ids)
    max_teacher_len = max(len(ids) for ids in all_teacher_ids)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    student_input_ids = torch.full((batch_size, max_student_len), pad_id, dtype=torch.long)
    student_position_ids = torch.zeros(batch_size, max_student_len, dtype=torch.long)
    student_completion_mask = torch.zeros(batch_size, max_student_len, dtype=torch.bool)
    student_labels = torch.full((batch_size, max_student_len), -100, dtype=torch.long)

    teacher_input_ids = torch.full((batch_size, max_teacher_len), pad_id, dtype=torch.long)
    teacher_position_ids = torch.zeros(batch_size, max_teacher_len, dtype=torch.long)
    teacher_completion_mask = torch.zeros(batch_size, max_teacher_len, dtype=torch.bool)

    for i in range(batch_size):
        s_ids = all_student_ids[i]
        t_ids = all_teacher_ids[i]
        comp_len = all_completion_lengths[i]

        # Left-pad student
        s_pad = max_student_len - len(s_ids)
        student_input_ids[i, s_pad:] = torch.tensor(s_ids, dtype=torch.long)
        student_position_ids[i, s_pad:] = torch.arange(len(s_ids))

        # Completion mask: last comp_len tokens
        if comp_len > 0:
            start = max_student_len - comp_len + num_loss_tokens_to_skip
            student_completion_mask[i, start:] = True

        # Labels: shifted input_ids (next-token prediction on completion only)
        if comp_len > 1:
            label_start = max_student_len - comp_len
            student_labels[i, label_start : max_student_len - 1] = student_input_ids[i, label_start + 1 : max_student_len]

        # Left-pad teacher
        t_pad = max_teacher_len - len(t_ids)
        teacher_input_ids[i, t_pad:] = torch.tensor(t_ids, dtype=torch.long)
        teacher_position_ids[i, t_pad:] = torch.arange(len(t_ids))

        if comp_len > 0:
            t_start = max_teacher_len - comp_len + num_loss_tokens_to_skip
            teacher_completion_mask[i, t_start:] = True

    return SDFTTrainBatch(
        student_input_ids=student_input_ids,
        student_position_ids=student_position_ids,
        teacher_input_ids=teacher_input_ids,
        teacher_position_ids=teacher_position_ids,
        completion_mask=student_completion_mask,
        teacher_completion_mask=teacher_completion_mask,
        labels=student_labels,
    )
