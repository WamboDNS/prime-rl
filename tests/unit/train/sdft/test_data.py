import pytest
from transformers import AutoTokenizer

from prime_rl.trainer.sdft.data import FakeSDFTDataset, prepare_sdft_batch


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_fake_dataset_yields_batches():
    """FakeSDFTDataset yields batches with student and teacher prompts."""
    dataset = FakeSDFTDataset(batch_size=4)
    batch = next(iter(dataset))
    assert "student_prompts" in batch
    assert "teacher_prompts" in batch
    assert len(batch["student_prompts"]) == 4
    assert len(batch["teacher_prompts"]) == 4


def test_fake_dataset_prompts_differ():
    """Student and teacher prompts are different."""
    dataset = FakeSDFTDataset(batch_size=2)
    batch = next(iter(dataset))
    for s, t in zip(batch["student_prompts"], batch["teacher_prompts"]):
        assert s != t
        assert len(t) > len(s)  # Teacher prompt includes example


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


def test_completion_mask_alignment(tokenizer):
    """Completion mask correctly identifies completion tokens."""
    student_prompts = ["Hello"]
    teacher_prompts = ["Hello, here is an example: Hi!\n\nNow you:"]
    completions = ["World"]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        tokenizer=tokenizer,
        max_prompt_length=128,
        max_completion_length=64,
    )

    # Both masks should have the same number of True values
    student_comp_count = batch["completion_mask"].sum().item()
    teacher_comp_count = batch["teacher_completion_mask"].sum().item()
    assert student_comp_count == teacher_comp_count
    assert student_comp_count > 0


def test_position_ids_correct(tokenizer):
    """Position IDs are correct for padded sequences."""
    student_prompts = ["Short", "A longer prompt here"]
    teacher_prompts = ["Short example", "A longer prompt here with example"]
    completions = ["answer1", "answer2"]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        tokenizer=tokenizer,
        max_prompt_length=128,
        max_completion_length=64,
    )

    # Position IDs should start at 0 for actual content (after left padding)
    for i in range(2):
        pos_ids = batch["student_position_ids"][i]
        non_zero_mask = pos_ids > 0
        if non_zero_mask.any():
            first_nonzero = non_zero_mask.nonzero()[0].item()
            assert pos_ids[first_nonzero].item() >= 1


def test_truncation(tokenizer):
    """Long prompts and completions are truncated."""
    student_prompts = ["x " * 1000]
    teacher_prompts = ["y " * 1000]
    completions = ["z " * 1000]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        tokenizer=tokenizer,
        max_prompt_length=32,
        max_completion_length=16,
    )

    assert batch["student_input_ids"].shape[1] <= 32 + 16
    assert batch["teacher_input_ids"].shape[1] <= 32 + 16


def test_chat_template_applied(tokenizer):
    """Chat template tokens are present in tokenized output."""
    student_prompts = ["What is 2+2?"]
    teacher_prompts = ["What is 2+2?\n\nExample: 4\n\nNow answer:"]
    completions = ["The answer is 4."]

    batch = prepare_sdft_batch(
        student_prompts=student_prompts,
        teacher_prompts=teacher_prompts,
        completions=completions,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_completion_length=64,
    )

    # The chat template should add special tokens (e.g. <|im_start|>)
    # that wouldn't be present if we just tokenized the raw string.
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": student_prompts[0]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    chat_only_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    raw_only_ids = tokenizer.encode(student_prompts[0], add_special_tokens=False)

    # Chat template should produce more tokens than raw encoding
    assert len(chat_only_ids) > len(raw_only_ids)

    # The student_input_ids should contain the chat template tokens, not just the raw ones
    student_ids = batch["student_input_ids"][0].tolist()
    # Chat template IDs should appear as a subsequence in the student input
    assert all(tid in student_ids for tid in chat_only_ids)
