from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch import Tensor
from torch.utils.data import IterableDataset
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sdft.config import SDFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


class SDFTPromptBatch(TypedDict):
    """A batch of prompts with ground truth answers for SDFT."""

    prompts: list[str]
    answers: list[str]
    systems: list[str | None]
    kinds: list[str | None]


class SDFTTrainBatch(TypedDict):
    """A tokenized training batch for SDFT with dual inputs."""

    student_input_ids: Tensor  # [batch, seq]
    student_position_ids: Tensor  # [batch, seq]
    teacher_input_ids: Tensor  # [batch, seq]
    teacher_position_ids: Tensor  # [batch, seq]
    completion_mask: Tensor  # [batch, seq] (aligned with student)
    teacher_completion_mask: Tensor  # [batch, seq] (aligned with teacher)
    labels: Tensor  # [batch, seq] target token ids
    self_distillation_mask: Tensor  # [batch] whether this sample has a demonstration


class SDFTDataset(IterableDataset):
    """Dataset that yields prompt batches for SDFT training."""

    def __init__(
        self,
        dataset: Dataset,
        prompt_field: str = "prompt",
        answer_field: str = "answer",
        system_field: str | None = "system",
        kind_field: str | None = "kind",
        shuffle: bool = True,
        seed: int = 0,
        batch_size: int = 4,
    ):
        self.logger = get_logger()
        self.dataset = dataset
        self.prompt_field = prompt_field
        self.answer_field = answer_field
        self.system_field = system_field
        self.kind_field = kind_field
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

            batch_prompts = []
            batch_answers = []
            batch_systems = []
            batch_kinds = []

            while len(batch_prompts) < self.batch_size:
                idx = self.step % self.num_examples

                if self.step % self.data_world_size == self.data_rank:
                    example = dataset[idx]
                    batch_prompts.append(example[self.prompt_field])
                    batch_answers.append(example[self.answer_field])
                    batch_systems.append(example.get(self.system_field) if self.system_field else None)
                    batch_kinds.append(example.get(self.kind_field) if self.kind_field else None)

                self.step += 1
                epoch = self.step // self.num_examples
                if epoch > self.epoch:
                    self.epoch = epoch
                    dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            yield SDFTPromptBatch(
                prompts=batch_prompts,
                answers=batch_answers,
                systems=batch_systems,
                kinds=batch_kinds,
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
            prompts = [f"What is {i}+{i}?" for i in range(self.step, self.step + self.batch_size)]
            answers = [str(2 * i) for i in range(self.step, self.step + self.batch_size)]
            self.step += self.batch_size
            yield SDFTPromptBatch(
                prompts=prompts,
                answers=answers,
                systems=[None] * self.batch_size,
                kinds=[None] * self.batch_size,
            )


def setup_sdft_dataset(config: SDFTDataConfig, num_prompts_per_batch: int) -> SDFTDataset | FakeSDFTDataset:
    if config.type == "fake":
        return FakeSDFTDataset(batch_size=num_prompts_per_batch)

    logger = get_logger()
    logger.info(f"Loading SDFT dataset: {config.dataset_name} (split={config.dataset_split})")

    path = Path(config.dataset_name)
    if path.is_dir() and (path / "state.json").exists():
        dataset = cast(Dataset, load_from_disk(config.dataset_name))
    elif path.is_dir():
        # Directory with raw JSON/Parquet files (not HF arrow format)
        json_files = list(path.glob("train.json")) + list(path.glob("train.jsonl"))
        parquet_files = list(path.glob("train.parquet"))
        if json_files:
            dataset = cast(Dataset, load_dataset("json", data_files=str(json_files[0]), split="train"))
        elif parquet_files:
            dataset = cast(Dataset, load_dataset("parquet", data_files=str(parquet_files[0]), split="train"))
        else:
            raise FileNotFoundError(f"Directory {path} has no train.json, train.jsonl, or train.parquet files")
    elif path.exists() and path.suffix in (".json", ".jsonl"):
        dataset = cast(Dataset, load_dataset("json", data_files=str(path), split="train"))
    elif path.exists() and path.suffix == ".parquet":
        dataset = cast(Dataset, load_dataset("parquet", data_files=str(path), split="train"))
    else:
        dataset = cast(Dataset, load_dataset(config.dataset_name, split=config.dataset_split))

    assert config.prompt_field in dataset.column_names, (
        f"Dataset must have a '{config.prompt_field}' column, found: {dataset.column_names}"
    )
    assert config.answer_field in dataset.column_names, (
        f"Dataset must have a '{config.answer_field}' column, found: {dataset.column_names}"
    )

    return SDFTDataset(
        dataset=dataset,
        prompt_field=config.prompt_field,
        answer_field=config.answer_field,
        system_field=config.system_field if config.system_field and config.system_field in dataset.column_names else None,
        kind_field=config.kind_field if config.kind_field and config.kind_field in dataset.column_names else None,
        shuffle=config.shuffle,
        seed=config.seed,
        batch_size=num_prompts_per_batch,
    )


def prepare_sdft_batch(
    student_prompts: list[str],
    teacher_prompts: list[str],
    completions: list[str],
    self_distillation_mask: list[bool],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    max_completion_length: int,
    max_reprompt_length: int | None = None,
    reprompt_truncation: Literal["left", "right", "error"] = "right",
    student_systems: list[str | None] | None = None,
    teacher_systems: list[str | None] | None = None,
) -> SDFTTrainBatch:
    """Tokenize dual prompts + shared completions into padded training tensors.

    Both student and teacher get the same completion appended. The completion_mask
    identifies which tokens are from the completion (where we compute KL loss).
    """
    batch_size = len(student_prompts)

    all_student_ids = []
    all_teacher_ids = []
    all_completion_lengths = []

    if student_systems is None:
        student_systems = [None] * batch_size
    if teacher_systems is None:
        teacher_systems = [None] * batch_size

    def truncate_prompt_ids(ids: list[int], max_len: int, mode: Literal["left", "right", "error"]) -> list[int]:
        if len(ids) <= max_len:
            return ids
        if mode == "left":
            return ids[-max_len:]
        if mode == "right":
            return ids[:max_len]
        raise ValueError(
            f"Teacher prompt length {len(ids)} exceeds max_reprompt_length={max_len} with reprompt_truncation='error'"
        )

    for i in range(batch_size):
        student_messages = []
        if student_systems[i]:
            student_messages.append({"role": "system", "content": student_systems[i]})
        student_messages.append({"role": "user", "content": student_prompts[i]})

        teacher_messages = []
        if teacher_systems[i]:
            teacher_messages.append({"role": "system", "content": teacher_systems[i]})
        teacher_messages.append({"role": "user", "content": teacher_prompts[i]})

        student_prompt_text = tokenizer.apply_chat_template(
            student_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        student_prompt_ids = tokenizer.encode(student_prompt_text, add_special_tokens=False)
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completions[i], add_special_tokens=False)

        student_prompt_ids = student_prompt_ids[-max_prompt_length:]
        if max_reprompt_length is not None:
            teacher_prompt_ids = truncate_prompt_ids(teacher_prompt_ids, max_reprompt_length, reprompt_truncation)
        else:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]
        completion_ids = completion_ids[:max_completion_length]

        all_student_ids.append(student_prompt_ids + completion_ids)
        all_teacher_ids.append(teacher_prompt_ids + completion_ids)
        all_completion_lengths.append(len(completion_ids))

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

    sd_mask = torch.tensor(self_distillation_mask, dtype=torch.float32)

    for i in range(batch_size):
        s_ids = all_student_ids[i]
        t_ids = all_teacher_ids[i]
        comp_len = all_completion_lengths[i]

        # Left-pad student
        s_pad = max_student_len - len(s_ids)
        student_input_ids[i, s_pad:] = torch.tensor(s_ids, dtype=torch.long)
        student_position_ids[i, s_pad:] = torch.arange(len(s_ids))

        if comp_len > 0:
            start = max_student_len - comp_len
            student_completion_mask[i, start:] = True

        if comp_len > 1:
            label_start = max_student_len - comp_len
            student_labels[i, label_start : max_student_len - 1] = student_input_ids[i, label_start + 1 : max_student_len]

        # Left-pad teacher
        t_pad = max_teacher_len - len(t_ids)
        teacher_input_ids[i, t_pad:] = torch.tensor(t_ids, dtype=torch.long)
        teacher_position_ids[i, t_pad:] = torch.arange(len(t_ids))

        if comp_len > 0:
            t_start = max_teacher_len - comp_len
            teacher_completion_mask[i, t_start:] = True

    return SDFTTrainBatch(
        student_input_ids=student_input_ids,
        student_position_ids=student_position_ids,
        teacher_input_ids=teacher_input_ids,
        teacher_position_ids=teacher_position_ids,
        completion_mask=student_completion_mask,
        teacher_completion_mask=teacher_completion_mask,
        labels=student_labels,
        self_distillation_mask=sd_mask,
    )
