"""Profile the SDFT training step to identify performance bottlenecks.

Replicates the exact training step from train.py (model setup, FSDP, fused
distillation path) with synthetic data. Profiles with PyTorch profiler.

Usage (3 GPUs matching generalization.toml trainer_gpu_ids):
    CUDA_VISIBLE_DEVICES=1,2,3 uv run torchrun --nproc-per-node=3 \
        scripts/sdft/profile_training.py

Single GPU (faster iteration):
    CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node=1 \
        scripts/sdft/profile_training.py
"""

import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from loguru import logger
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torchtitan.distributed.utils import clip_grad_norm_

from prime_rl.trainer.model import setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.sdft.config import SDFTTrainerConfig
from prime_rl.trainer.sdft.fused_distill import fused_distill_topk
from prime_rl.trainer.sdft.loss import add_tail, sdft_kl_loss
from prime_rl.trainer.sdft.train import (
    _compute_rollout_is_weights,
    _compute_token_log_probs_from_hidden,
    _extract_dense_completions,
    ema_update,
    forward_hidden,
    setup_replicated_teacher,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import setup_logger


PROFILE_CONFIG = dict(
    model_name="Qwen/Qwen3-8B",
    micro_batch_size=2,
    max_prompt_length=2048,
    max_completion_length=8192,
    distillation_topk=100,
    distillation_chunk_size=16384,
    alpha=0.5,
    is_clip=2.0,
    rollout_is="token",
    rollout_is_threshold=2.0,
    ema_update_rate=0.05,
    ac_freq=1,  # None = no AC, int = AC every N layers
    reshard_after_forward=False,  # False = keep params in memory after fwd
    num_warmup=2,
    num_profiled=3,
)


def make_synthetic_batch(micro_batch_size, student_seq_len, teacher_seq_len, vocab_size, completion_len):
    """Create a synthetic micro-batch matching the biology experiment's shapes."""
    student_input_ids = torch.randint(0, vocab_size, (micro_batch_size, student_seq_len), device="cuda")
    student_position_ids = torch.arange(student_seq_len, device="cuda").unsqueeze(0).expand(micro_batch_size, -1)

    teacher_input_ids = torch.randint(0, vocab_size, (micro_batch_size, teacher_seq_len), device="cuda")
    teacher_position_ids = torch.arange(teacher_seq_len, device="cuda").unsqueeze(0).expand(micro_batch_size, -1)

    # Completion mask: last `completion_len` tokens are completions (left-padded layout)
    completion_mask = torch.zeros(micro_batch_size, student_seq_len, dtype=torch.bool, device="cuda")
    completion_mask[:, -completion_len:] = True

    teacher_completion_mask = torch.zeros(micro_batch_size, teacher_seq_len, dtype=torch.bool, device="cuda")
    teacher_completion_mask[:, -completion_len:] = True

    sd_mask = torch.ones(micro_batch_size, dtype=torch.bool, device="cuda")

    old_token_log_probs = torch.randn(micro_batch_size, student_seq_len, device="cuda") * 0.1 - 5.0

    return {
        "student_input_ids": student_input_ids,
        "student_position_ids": student_position_ids,
        "teacher_input_ids": teacher_input_ids,
        "teacher_position_ids": teacher_position_ids,
        "completion_mask": completion_mask,
        "teacher_completion_mask": teacher_completion_mask,
        "self_distillation_mask": sd_mask,
        "old_token_log_probs": old_token_log_probs,
    }


def run_training_step(model, teacher_model, optimizer, micro_batch, cfg):
    """Execute one full training step using all v2 optimizations:
    - Dense completion extraction (no boolean indexing / nonzero syncs)
    - Larger chunk size for fused distillation
    - Replicated teacher (no FSDP allgather)
    """
    student_input_ids = micro_batch["student_input_ids"]
    student_position_ids = micro_batch["student_position_ids"]
    teacher_input_ids = micro_batch["teacher_input_ids"]
    teacher_position_ids = micro_batch["teacher_position_ids"]
    completion_mask = micro_batch["completion_mask"]
    teacher_completion_mask = micro_batch["teacher_completion_mask"]
    sd_mask = micro_batch["self_distillation_mask"]

    with record_function("student_forward_hidden"):
        student_hidden = forward_hidden(model, student_input_ids, student_position_ids)

    with record_function("teacher_forward_hidden"):
        with torch.no_grad():
            teacher_hidden = forward_hidden(teacher_model, teacher_input_ids, teacher_position_ids)

    with record_function("extract_completion_dense"):
        s_dense, s_valid_mask, max_comp_len = _extract_dense_completions(student_hidden, completion_mask)
        t_dense, t_valid_mask, _ = _extract_dense_completions(teacher_hidden, teacher_completion_mask)
        aligned_mask = s_valid_mask & sd_mask.unsqueeze(1).bool()

    B, _, H = s_dense.shape
    lm_weight = model.lm_head.weight

    with record_function("fused_distill_topk"):
        s_flat = s_dense.reshape(-1, H)
        t_flat = t_dense.reshape(-1, H)
        student_topk_lp, teacher_topk_lp, _ = fused_distill_topk(
            s_flat, t_flat, lm_weight,
            K=cfg["distillation_topk"],
            chunk_size=cfg["distillation_chunk_size"],
        )
        student_topk_lp = student_topk_lp.reshape(B, max_comp_len, -1)
        teacher_topk_lp = teacher_topk_lp.reshape(B, max_comp_len, -1)

    with record_function("add_tail"):
        student_distill_lp = add_tail(student_topk_lp)
        teacher_distill_lp = add_tail(teacher_topk_lp)

    with record_function("is_correction"):
        with torch.no_grad():
            student_token_lp = _compute_token_log_probs_from_hidden(
                hidden=student_hidden.detach(),
                input_ids=student_input_ids,
                lm_weight=lm_weight,
                chunk_size=cfg["distillation_chunk_size"],
            )
        student_token_lp_comp = student_token_lp[:, -max_comp_len:]
        old_token_lp_comp = micro_batch["old_token_log_probs"][:, -max_comp_len:]
        negative_approx_kl = (student_token_lp_comp - old_token_lp_comp).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        is_ratio = torch.exp(negative_approx_kl).clamp(max=cfg["is_clip"])
        rollout_is_weights = _compute_rollout_is_weights(
            log_ratio=negative_approx_kl,
            mask=aligned_mask,
            rollout_is=cfg["rollout_is"],
            threshold=cfg["rollout_is_threshold"],
        )

    with record_function("kl_loss"):
        loss, metrics = sdft_kl_loss(
            student_distill_lp,
            teacher_distill_lp,
            aligned_mask,
            alpha=cfg["alpha"],
            is_ratio=is_ratio,
            rollout_is_weights=rollout_is_weights,
        )

    del student_hidden, teacher_hidden
    del student_distill_lp, teacher_distill_lp

    with record_function("backward"):
        loss.backward()

    with record_function("grad_clip"):
        clip_grad_norm_(model.parameters(), max_norm=1.0, ep_enabled=False)

    with record_function("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad()

    # EMA skipped during profiling — in production it's amortized over
    # grad_accum_steps (16 micro-batches), not every micro-batch.

    return loss.item()


def main():
    cfg = PROFILE_CONFIG

    setup_logger("info")
    torch.set_float32_matmul_precision("high")

    from prime_rl.trainer.utils import setup_torch_distributed
    setup_torch_distributed(timeout=timedelta(seconds=120))

    world = get_world()
    logger.info(f"Starting SDFT profiler v3 on rank {world.rank}/{world.world_size}")
    logger.info(f"AC={cfg['ac_freq']}, reshard_after_forward={cfg['reshard_after_forward']}, replicated teacher, chunk_size={cfg['distillation_chunk_size']}")

    # Build trainer config
    model_cfg = {
        "name": cfg["model_name"],
        "fused_lm_head_chunk_size": "disabled",
        "reshard_after_forward": cfg["reshard_after_forward"],
    }
    if cfg["ac_freq"] is not None:
        model_cfg["ac"] = {"freq": cfg["ac_freq"]}

    trainer_config = SDFTTrainerConfig(
        model=model_cfg,
        data={"micro_batch_size": cfg["micro_batch_size"]},
        generation={"max_prompt_length": cfg["max_prompt_length"], "max_completion_length": cfg["max_completion_length"]},
    )

    max_seq_len = cfg["max_prompt_length"] + cfg["max_completion_length"]
    parallel_dims = get_parallel_dims(trainer_config.model, max_seq_len)

    logger.info("Loading FSDP student model...")
    model = setup_model(trainer_config.model, parallel_dims, False)

    logger.info("Loading replicated teacher model (no FSDP)...")
    teacher_model = setup_replicated_teacher(trainer_config.model)

    optimizer = setup_optimizer(trainer_config.optim, list(model.named_parameters()), parallel_dims)

    vocab_size = model.lm_head.weight.shape[0]

    # Synthetic batch — realistic completion length
    completion_len = 1024
    student_seq_len = cfg["max_prompt_length"] + completion_len
    teacher_seq_len = cfg["max_prompt_length"] + 2048 + completion_len

    logger.info(
        f"Synthetic batch: micro_bs={cfg['micro_batch_size']}, "
        f"student_seq={student_seq_len}, teacher_seq={teacher_seq_len}, "
        f"completion={completion_len}, vocab={vocab_size}"
    )

    micro_batch = make_synthetic_batch(
        cfg["micro_batch_size"], student_seq_len, teacher_seq_len, vocab_size, completion_len,
    )

    # Warmup
    logger.info(f"Running {cfg['num_warmup']} warmup steps...")
    for i in range(cfg["num_warmup"]):
        t0 = time.perf_counter()
        loss = run_training_step(model, teacher_model, optimizer, micro_batch, cfg)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        logger.info(f"Warmup step {i}: loss={loss:.4f}, time={dt:.2f}s")

    # Profiled steps — save to v3 directory to preserve v1/v2 results
    logger.info(f"Running {cfg['num_profiled']} profiled steps...")
    output_dir = "/home/ubuntu/prime-rl/outputs/profile_v3"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=cfg["num_profiled"], repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(cfg["num_profiled"]):
            t0 = time.perf_counter()
            loss = run_training_step(model, teacher_model, optimizer, micro_batch, cfg)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            logger.info(f"Profiled step {i}: loss={loss:.4f}, time={dt:.2f}s")
            prof.step()

    # Print summary tables (use sys.stdout.write + flush to ensure tee captures them)
    if world.is_master:
        tables = []
        tables.append("=== CUDA Time Summary (top 30) ===")
        tables.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        tables.append("")
        tables.append("=== CPU Time Summary (top 30) ===")
        tables.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        tables.append("")
        tables.append("=== By record_function label ===")
        tables.append(prof.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total", row_limit=50,
        ))
        output = "\n".join(tables)
        sys.stderr.write(output + "\n")
        sys.stderr.flush()

        logger.success(f"Profile v3 trace saved to {output_dir}/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
