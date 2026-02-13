import asyncio
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from loguru import logger
from openai import AsyncOpenAI
from torchtitan.distributed.utils import clip_grad_norm_

from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.model import forward, setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.sdft.config import SDFTTrainerConfig
from prime_rl.trainer.sdft.data import (
    prepare_sdft_batch,
    setup_sdft_dataset,
)
from prime_rl.trainer.sdft.loss import entropy_mask, sdft_kl_loss
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.weights import gather_weights_on_master, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, get_broadcast_dir, get_step_path


async def generate_completions(
    client: AsyncOpenAI,
    prompts: list[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    """Generate completions from the inference server using the OpenAI API."""
    tasks = []
    for prompt in prompts:
        task = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body={"top_k": top_k},
        )
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content or "" for r in responses]


def broadcast_weights_to_inference(model, output_dir, step, world):
    """Save model weights to filesystem so the inference server can pick them up."""
    broadcast_dir = get_broadcast_dir(output_dir)
    state_dict = gather_weights_on_master(model, is_master=world.is_master)
    if world.is_master:
        step_path = get_step_path(broadcast_dir, step)
        save_state_dict(state_dict, step_path)
        # Write a marker file so the inference server knows new weights are available
        (broadcast_dir / "latest_step.txt").write_text(str(step))
    dist.barrier()


def ema_sync(ref_model, student_model, alpha):
    """Sync reference model with student: ref = alpha*student + (1-alpha)*ref."""
    with torch.no_grad():
        for ref_param, student_param in zip(ref_model.parameters(), student_model.parameters()):
            ref_param.data.mul_(1 - alpha).add_(student_param.data, alpha=alpha)


@clean_exit
@logger.catch(reraise=True)
def train(config: SDFTTrainerConfig):
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
    )
    logger.info(f"Starting SDFT trainer in {world}")

    # Setup monitor
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Setup distributed
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision("high")

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model, config.generation.max_prompt_length + config.generation.max_completion_length)

    grad_accum_steps = config.data.batch_size // config.data.micro_batch_size

    # Setup checkpoint managers
    ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt, config.model.lora)

    checkpoint_step = None
    if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            checkpoint_step = config.ckpt.resume_step

    # Initialize model
    logger.info(f"Initializing model ({config.model.name})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)

    # Initialize tokenizer
    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Optional: setup EMA reference model
    ref_model = None
    if config.ref_model.enabled:
        logger.info("Initializing EMA reference model")
        ref_model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
        for param in ref_model.parameters():
            param.requires_grad = False

    # Setup optimizer
    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, list(model.named_parameters()), parallel_dims)

    # Setup scheduler
    scheduler_steps = config.max_steps
    logger.info(f"Setting up {config.scheduler.type} scheduler with {scheduler_steps} steps")
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    # Setup dataset
    logger.info(f"Initializing dataset ({config.data})")
    dataset = setup_sdft_dataset(config.data)
    dataset_iter = iter(dataset)

    # Resume from checkpoint if needed
    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")

    # Setup inference client
    api_key = os.environ.get("VLLM_API_KEY", "dummy")
    base_url = config.client.base_url[0]
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=config.client.timeout)

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3

    buffer = []
    buffer_idx = 0

    while True:
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        # Checkpoint saving
        if (
            ckpt_manager is not None
            and config.ckpt
            and config.ckpt.interval
            and progress.step > 0
            and not is_last_step
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
            ckpt_manager.maybe_clean()
            if weight_ckpt_manager is not None:
                weight_ckpt_manager.save(progress.step, model, tokenizer)
                weight_ckpt_manager.maybe_clean()

        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        # === Generation phase ===
        if not buffer or buffer_idx >= len(buffer):
            torch.cuda.reset_peak_memory_stats()
            step_start_time = time.perf_counter()

            # Get next batch of dual prompts
            prompt_batch = next(dataset_iter)
            student_prompts = prompt_batch["student_prompts"]
            teacher_prompts = prompt_batch["teacher_prompts"]

            # Choose which prompt to generate from
            gen_prompts = teacher_prompts if config.generation.generate_from_teacher else student_prompts

            # Generate completions from inference server (only rank 0, then broadcast)
            if world.is_master:
                logger.info(f"Generating {len(gen_prompts)} completions...")
                completions = asyncio.get_event_loop().run_until_complete(
                    generate_completions(
                        client=client,
                        prompts=gen_prompts,
                        model_name=config.model.name,
                        max_tokens=config.generation.max_completion_length,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        top_k=config.generation.top_k,
                    )
                )
            else:
                completions = None

            # Broadcast completions to all ranks
            if world.world_size > 1:
                import pickle

                if world.is_master:
                    data = pickle.dumps(completions)
                    size_tensor = torch.tensor([len(data)], dtype=torch.long, device="cuda")
                else:
                    size_tensor = torch.tensor([0], dtype=torch.long, device="cuda")
                dist.broadcast(size_tensor, src=0)

                if world.is_master:
                    data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to("cuda")
                else:
                    data_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cuda")
                dist.broadcast(data_tensor, src=0)

                if not world.is_master:
                    completions = pickle.loads(data_tensor.cpu().numpy().tobytes())

            gen_time = time.perf_counter() - step_start_time
            logger.debug(f"Generation took {gen_time:.2f}s")

            # Tokenize and prepare training batch
            train_batch = prepare_sdft_batch(
                student_prompts=student_prompts,
                teacher_prompts=teacher_prompts,
                completions=completions,
                tokenizer=tokenizer,
                max_prompt_length=config.generation.max_prompt_length,
                max_completion_length=config.generation.max_completion_length,
                num_loss_tokens_to_skip=config.loss.num_loss_tokens_to_skip,
            )

            # Split into micro-batches for gradient accumulation
            buffer = []
            bs = config.data.micro_batch_size
            for mb_idx in range(0, config.data.batch_size, bs):
                mb = {
                    k: v[mb_idx : mb_idx + bs] for k, v in train_batch.items()
                }
                buffer.append(mb)

            # Repeat buffer for num_iterations
            if config.generation.num_iterations > 1:
                buffer = buffer * config.generation.num_iterations

            buffer_idx = 0

            # Broadcast weights to inference server after generation
            broadcast_weights_to_inference(model, config.output_dir, progress.step, world)

        # === Training phase ===
        train_step_start = time.perf_counter()
        batch_loss = torch.tensor(0.0, device="cuda")
        batch_kl = torch.tensor(0.0, device="cuda")

        for micro_step in range(grad_accum_steps):
            if buffer_idx >= len(buffer):
                break
            micro_batch = buffer[buffer_idx]
            buffer_idx += 1

            # Move to device
            student_input_ids = micro_batch["student_input_ids"].to("cuda")
            student_position_ids = micro_batch["student_position_ids"].to("cuda")
            teacher_input_ids = micro_batch["teacher_input_ids"].to("cuda")
            teacher_position_ids = micro_batch["teacher_position_ids"].to("cuda")
            completion_mask = micro_batch["completion_mask"].to("cuda")
            teacher_completion_mask = micro_batch["teacher_completion_mask"].to("cuda")

            # Student forward pass (with gradients)
            student_out = forward(model, student_input_ids, student_position_ids)
            student_logits = student_out["logits"]

            # Teacher forward pass (no gradients)
            teacher_model = ref_model if ref_model is not None else model
            with torch.no_grad():
                teacher_out = forward(teacher_model, teacher_input_ids, teacher_position_ids)
                teacher_logits = teacher_out["logits"]

            # Align logits to completion tokens
            # Student logits shape: [batch, student_seq, vocab]
            # Teacher logits shape: [batch, teacher_seq, vocab]
            # We need to extract only completion logits from both (they have same completion length)
            student_comp_logits = _extract_completion_logits(student_logits, completion_mask)
            teacher_comp_logits = _extract_completion_logits(teacher_logits, teacher_completion_mask)

            # Build aligned completion mask (all True since we extracted exactly completion tokens)
            comp_len = student_comp_logits.shape[1]
            aligned_mask = torch.ones(student_comp_logits.shape[0], comp_len, dtype=torch.bool, device="cuda")

            # Optional entropy masking
            if config.loss.top_entropy_quantile < 1.0:
                aligned_mask = entropy_mask(
                    teacher_comp_logits, aligned_mask, config.loss.top_entropy_quantile
                )

            # Compute KL loss
            loss, metrics = sdft_kl_loss(
                student_comp_logits,
                teacher_comp_logits,
                aligned_mask,
                alpha=config.loss.alpha,
                temperature=config.loss.temperature,
            )

            batch_loss += loss.detach() / grad_accum_steps
            batch_kl += metrics["kl_divergence"] / grad_accum_steps

            # Delete logits before backward to save memory
            del student_logits, teacher_logits, student_comp_logits, teacher_comp_logits

            # Backward
            (loss / grad_accum_steps).backward()

        # Optimizer step
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm, ep_enabled=False)
        if grad_norm.device.type == "cpu":
            grad_norm = grad_norm.to("cuda")

        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Optional EMA sync
        if ref_model is not None and (progress.step + 1) % config.ref_model.sync_steps == 0:
            ema_sync(ref_model, model, config.ref_model.mixup_alpha)

        # Synchronize metrics
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_kl, op=dist.ReduceOp.AVG)

        # Log metrics
        step_time = time.perf_counter() - train_step_start
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3

        step_message = (
            f"Step {progress.step} | "
            f"Time: {step_time:.2f}s | "
            f"KL Loss: {batch_loss.item():.4f} | "
            f"KL Div: {batch_kl.item():.4f} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Peak Mem: {peak_memory:.1f}/{max_memory:.1f} GiB"
        )
        logger.success(step_message)

        monitor.log(
            {
                "loss/kl": batch_loss.item(),
                "loss/kl_divergence": batch_kl.item(),
                "optim/lr": current_lr,
                "optim/grad_norm": grad_norm.item(),
                "perf/peak_memory": peak_memory,
                "time/step": step_time,
                "step": progress.step,
            },
            step=progress.step,
        )

        progress.step += 1

    # Final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
        ckpt_manager.maybe_clean()
    if weight_ckpt_manager is not None:
        weight_ckpt_manager.save(progress.step, model, tokenizer)
        weight_ckpt_manager.maybe_clean()

    logger.success("SDFT trainer finished!")


def _extract_completion_logits(logits: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    """Extract logits corresponding to completion tokens and stack into a dense tensor.

    Args:
        logits: [batch, seq, vocab]
        completion_mask: [batch, seq] bool

    Returns:
        [batch, max_comp_len, vocab] with zero-padding where needed.
    """
    batch_size = logits.shape[0]
    max_comp_len = completion_mask.sum(dim=-1).max().item()
    if max_comp_len == 0:
        return logits[:, :1, :]  # Fallback: return at least one token

    result = torch.zeros(batch_size, max_comp_len, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    for i in range(batch_size):
        comp_indices = completion_mask[i].nonzero(as_tuple=True)[0]
        n = comp_indices.shape[0]
        result[i, :n] = logits[i, comp_indices]
    return result


def main():
    train(parse_argv(SDFTTrainerConfig))


if __name__ == "__main__":
    main()
