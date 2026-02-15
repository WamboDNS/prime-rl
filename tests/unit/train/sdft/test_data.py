import pytest
from transformers import AutoTokenizer

from prime_rl.trainer.sdft.data import FakeSDFTDataset, prepare_sdft_batch


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_fake_dataset_yields_batches():
    """FakeSDFTDataset yields batches with prompts and answers."""
    dataset = FakeSDFTDataset(batch_size=4)
    batch = next(iter(dataset))
    assert "prompts" in batch
    assert "answers" in batch
    assert len(batch["prompts"]) == 4
    assert len(batch["answers"]) == 4


def test_prepare_batch_shapes(tokenizer):
    """prepare_sdft_batch produces tensors with correct shapes."""
    student_prompts = ["What is 2+2?", "What is 3+3?"]
    teacher_prompts = [
        "What is 2+2?\n\nExample: 4\n\nNow answer:",
        "What is 3+3?\n\nExample: 6\n\nNow answer:",
    ]
    completions = ["4", "6"]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        self_distillation_mask=[True, True],
        tokenizer=tokenizer,
        max_prompt_length=128,
        max_completion_length=64,
    )

    assert batch["student_input_ids"].shape[0] == 2
    assert batch["teacher_input_ids"].shape[0] == 2
    assert batch["student_input_ids"].ndim == 2
    assert batch["teacher_input_ids"].ndim == 2
    assert batch["completion_mask"].shape == batch["student_input_ids"].shape
    assert batch["teacher_completion_mask"].shape == batch["teacher_input_ids"].shape
    assert batch["self_distillation_mask"].shape == (2,)


def test_completion_mask_alignment(tokenizer):
    """Completion mask correctly identifies completion tokens."""
    student_prompts = ["Hello"]
    teacher_prompts = ["Hello, here is an example: Hi!\n\nNow you:"]
    completions = ["World"]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        self_distillation_mask=[True],
        tokenizer=tokenizer,
        max_prompt_length=128,
        max_completion_length=64,
    )

    student_comp_count = batch["completion_mask"].sum().item()
    teacher_comp_count = batch["teacher_completion_mask"].sum().item()
    assert student_comp_count == teacher_comp_count
    assert student_comp_count > 0


def test_self_distillation_mask(tokenizer):
    """self_distillation_mask controls which samples are trained."""
    student_prompts = ["Q1", "Q2"]
    teacher_prompts = ["Q1 with demo", "Q2 no demo"]
    completions = ["A1", "A2"]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        self_distillation_mask=[True, False],
        tokenizer=tokenizer,
        max_prompt_length=128,
        max_completion_length=64,
    )

    assert batch["self_distillation_mask"][0].item() == 1.0
    assert batch["self_distillation_mask"][1].item() == 0.0


def test_truncation(tokenizer):
    """Long prompts and completions are truncated."""
    student_prompts = ["x " * 1000]
    teacher_prompts = ["y " * 1000]
    completions = ["z " * 1000]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        self_distillation_mask=[True],
        tokenizer=tokenizer,
        max_prompt_length=32,
        max_completion_length=16,
    )

    assert batch["student_input_ids"].shape[1] <= 32 + 16
    assert batch["teacher_input_ids"].shape[1] <= 32 + 16
