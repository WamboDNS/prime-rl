import asyncio
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from openai import AsyncOpenAI
from torchtitan.distributed.utils import clip_grad_norm_

from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.model import forward, setup_model, setup_tokenizer
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.sdft.config import SDFTRepromptConfig, SDFTTrainerConfig
from prime_rl.trainer.sdft.data import (
    prepare_sdft_batch,
    setup_sdft_dataset,
)
from prime_rl.trainer.sdft.fused_distill import (
    chunked_token_log_probs_from_hidden,
    fused_distill_topk,
)
from prime_rl.trainer.sdft.loss import add_tail, renorm_topk_log_probs, sdft_kl_loss
from prime_rl.trainer.sdft.scoring import score_completion
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.weights import gather_weights_on_master, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.act_offloading import maybe_activation_offloading
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, get_broadcast_dir, get_step_path


async def generate_completions(
    client: AsyncOpenAI,
    prompts: list[str],
    system_messages: list[str | None],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_completions: int,
) -> list[str]:
    """Generate num_completions per prompt via the inference server."""
    tasks = []
    for prompt, system in zip(prompts, system_messages):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        tasks.append(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_completions,
                extra_body={"top_k": top_k},
            )
        )
    responses = await asyncio.gather(*tasks)
    # Flatten: each response has num_completions choices, emit in prompt-major order
    return [
        choice.message.content or ""
        for r in responses
        for choice in r.choices
    ]


def _remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def build_teacher_prompts(
    prompts: list[str],
    completions: list[str],
    scores: list[dict],
    num_completions: int,
    reprompt_cfg: SDFTRepromptConfig,
) -> tuple[list[str], list[bool]]:
    """Build teacher prompts with successful demonstrations (SDPO-style).

    Groups completions by prompt, finds successful ones, and builds reprompt
    messages. Returns teacher_prompts and self_distillation_mask.
    """
    num_prompts = len(prompts)
    batch_size = num_prompts * num_completions

    # Group successful completion indices by prompt
    success_by_prompt: dict[int, list[int]] = defaultdict(list)
    for i in range(batch_size):
        if scores[i]["score"] >= reprompt_cfg.success_threshold:
            success_by_prompt[i // num_completions].append(i)

    teacher_prompts = []
    sd_mask = []

    for i in range(batch_size):
        prompt_idx = i // num_completions
        prompt = prompts[prompt_idx]

        # Find a successful demonstration (not self if configured)
        solution_idxs = list(success_by_prompt[prompt_idx])
        if reprompt_cfg.dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != i]

        has_solution = len(solution_idxs) > 0
        feedback = scores[i].get("feedback") if reprompt_cfg.include_feedback else None
        has_feedback = feedback is not None and isinstance(feedback, str) and bool(feedback.strip())
        use_feedback = has_feedback and (not reprompt_cfg.environment_feedback_only_without_solution or not has_solution)

        if has_solution or use_feedback:
            solution_section = ""
            if has_solution:
                solution_text = completions[solution_idxs[0]]
                if reprompt_cfg.remove_thinking:
                    solution_text = _remove_thinking(solution_text)
                solution_section = reprompt_cfg.solution_template.format(
                    successful_previous_attempt=solution_text,
                )

            feedback_section = ""
            if use_feedback:
                feedback_section = reprompt_cfg.feedback_template.format(
                    feedback_raw=feedback,
                )

            reprompt_text = reprompt_cfg.reprompt_template.format(
                prompt=prompt,
                solution=solution_section,
                feedback=feedback_section,
            )
            teacher_prompts.append(reprompt_text)
            sd_mask.append(True)
        else:
            teacher_prompts.append(prompt)
            sd_mask.append(False)

    return teacher_prompts, sd_mask


def broadcast_weights_to_inference(model, output_dir, step, world):
    """Save model weights to filesystem so the inference server can pick them up."""
    broadcast_dir = get_broadcast_dir(output_dir)
    state_dict = gather_weights_on_master(model, is_master=world.is_master)
    if world.is_master:
        step_path = get_step_path(broadcast_dir, step)
        save_state_dict(state_dict, step_path)
        (broadcast_dir / "latest_step.txt").write_text(str(step))
        for old_dir in broadcast_dir.iterdir():
            if old_dir.is_dir() and old_dir != step_path:
                shutil.rmtree(old_dir)
    dist.barrier()


def ema_update(teacher_model, student_model, update_rate):
    """EMA update: teacher = rate*student + (1-rate)*teacher."""
    def local_tensor_or_self(tensor):
        if hasattr(tensor, "to_local"):
            return tensor.to_local()
        if hasattr(tensor, "_local_tensor"):
            return tensor._local_tensor
        return tensor

    student_params = dict(student_model.named_parameters())
    with torch.no_grad():
        for name, teacher_param in teacher_model.named_parameters():
            student_param = student_params[name]
            teacher_local = local_tensor_or_self(teacher_param.data)
            student_local = local_tensor_or_self(student_param.data)

            if student_local.shape == teacher_local.shape:
                student_tensor = student_local
            elif hasattr(student_param.data, "full_tensor"):
                student_tensor = student_param.data.full_tensor()
            else:
                raise RuntimeError(
                    f"EMA shape mismatch for {name}: teacher={tuple(teacher_local.shape)} "
                    f"student={tuple(student_local.shape)}"
                )

            teacher_local.mul_(1 - update_rate).add_(student_tensor, alpha=update_rate)


def _extract_completion_logits(logits: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    """Extract logits at completion positions into a dense tensor.

    Args:
        logits: [batch, seq, vocab]
        completion_mask: [batch, seq] bool

    Returns:
        [batch, max_comp_len, vocab] with zero-padding.
    """
    batch_size = logits.shape[0]
    max_comp_len = completion_mask.sum(dim=-1).max().item()
    if max_comp_len == 0:
        return logits[:, :1, :]

    result = torch.zeros(batch_size, max_comp_len, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    for i in range(batch_size):
        comp_indices = completion_mask[i].nonzero(as_tuple=True)[0]
        n = comp_indices.shape[0]
        result[i, :n] = logits[i, comp_indices]
    return result


def _compute_token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probability of each generated token.

    logits[t] predicts input_ids[t+1], so we shift and gather.

    Args:
        logits: [batch, seq, vocab]
        input_ids: [batch, seq]

    Returns:
        [batch, seq] with log_prob[t] = log π(input_ids[t+1] | ...) for t < seq-1,
        and 0.0 at the last position.
    """
    log_probs = logits[:, :-1, :].log_softmax(dim=-1)
    token_lps = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return F.pad(token_lps, (1, 0), value=0.0)


def _compute_token_log_probs_from_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    lm_weight: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute per-token log probabilities from hidden states in chunked fashion.

    Avoids materializing [batch, seq, vocab] logits for IS correction in fused SDFT.
    """
    batch_size, seq_len, hidden_dim = hidden.shape
    if seq_len <= 1:
        return torch.zeros(batch_size, seq_len, device=hidden.device, dtype=torch.float32)

    hidden_flat = hidden[:, :-1, :].reshape(-1, hidden_dim).contiguous()
    labels_flat = input_ids[:, 1:].reshape(-1).contiguous()
    token_lps_flat = chunked_token_log_probs_from_hidden(
        hidden=hidden_flat,
        next_token_ids=labels_flat,
        weight=lm_weight,
        chunk_size=chunk_size,
    )
    token_lps = token_lps_flat.reshape(batch_size, seq_len - 1)
    return F.pad(token_lps, (1, 0), value=0.0)


def _compute_rollout_is_weights(
    log_ratio: torch.Tensor,
    mask: torch.Tensor,
    rollout_is: str,
    threshold: float,
) -> torch.Tensor:
    """Compute rollout-correction IS weights from log-ratio and mask."""
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    if rollout_is == "token":
        weights = torch.exp(log_ratio)
    elif rollout_is == "sequence":
        seq_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True)
        weights = torch.exp(seq_log_ratio).expand_as(log_ratio)
    else:
        raise ValueError(f"Unsupported rollout_is mode: {rollout_is}")
    return weights.clamp(max=threshold).detach()


def _estimate_logits_memory_gib(tokens: int, vocab_size: int, bytes_per_elem: int) -> float:
    return (tokens * vocab_size * bytes_per_elem) / (1024**3)


def _completion_token_mask(completion_mask: torch.Tensor) -> torch.Tensor:
    """Build [batch, max_completion_len] mask for valid completion tokens."""
    lengths = completion_mask.sum(dim=-1)
    max_len = int(lengths.max().item())
    if max_len == 0:
        return torch.zeros(completion_mask.shape[0], 1, dtype=torch.bool, device=completion_mask.device)
    positions = torch.arange(max_len, device=completion_mask.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _scatter_flat_by_mask(flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scatter flattened valid-token tensor back to [batch, max_len, ...] padded form."""
    out_shape = (mask.shape[0], mask.shape[1], *flat.shape[1:])
    out = torch.zeros(out_shape, device=flat.device, dtype=flat.dtype)
    out[mask] = flat
    return out


def forward_hidden(model: torch.nn.Module, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Get hidden states without computing lm_head logits.

    Calls the full model forward (respecting FSDP's single-root constraint) but
    temporarily replaces lm_head.forward with a passthrough that captures hidden
    states without the expensive [N, V] matmul.
    """
    hidden_ref: list[torch.Tensor] = []
    original_forward = model.lm_head.forward

    def capture_hidden(hidden_states, labels=None, temperature=None):
        hidden_ref.append(hidden_states)
        return PrimeLmOutput()

    model.lm_head.forward = capture_hidden
    model(input_ids=input_ids, position_ids=position_ids)
    model.lm_head.forward = original_forward
    return hidden_ref[0]


def _extract_completion_values(values: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    """Extract per-token values at completion positions into a dense tensor.

    Args:
        values: [batch, seq]
        completion_mask: [batch, seq] bool

    Returns:
        [batch, max_comp_len] with zero-padding.
    """
    batch_size = values.shape[0]
    max_comp_len = completion_mask.sum(dim=-1).max().item()
    if max_comp_len == 0:
        return values[:, :1]

    result = torch.zeros(batch_size, max_comp_len, device=values.device, dtype=values.dtype)
    for i in range(batch_size):
        comp_indices = completion_mask[i].nonzero(as_tuple=True)[0]
        n = comp_indices.shape[0]
        result[i, :n] = values[i, comp_indices]
    return result


@clean_exit
@logger.catch(reraise=True)
def train(config: SDFTTrainerConfig):
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
    )
    logger.info(f"Starting SDFT trainer in {world}")

    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision("high")

    parallel_dims = get_parallel_dims(config.model, config.generation.max_prompt_length + config.generation.max_completion_length)

    mini_batch_size = config.data.mini_batch_size
    grad_accum_steps = mini_batch_size // config.data.micro_batch_size
    logger.info(
        f"Batch config: batch_size={config.data.batch_size}, mini_batch_size={mini_batch_size}, "
        f"micro_batch_size={config.data.micro_batch_size}, grad_accum_steps={grad_accum_steps}, "
        f"optimizer_steps_per_batch={config.data.batch_size // mini_batch_size}"
    )

    ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt, config.model.lora)

    checkpoint_step = None
    if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            checkpoint_step = config.ckpt.resume_step

    logger.info(f"Initializing model ({config.model.name})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Teacher model
    teacher_model = None
    if config.ref_model.enabled:
        logger.info(f"Initializing reference teacher model ({config.ref_model.regularization})")
        teacher_model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
        for param in teacher_model.parameters():
            param.requires_grad = False

    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, list(model.named_parameters()), parallel_dims)

    num_prompts_per_batch = config.data.batch_size // config.generation.num_completions
    logger.info(f"Initializing dataset ({config.data})")
    dataset = setup_sdft_dataset(config.data, num_prompts_per_batch)
    dataset_iter = iter(dataset)

    # Compute max_steps from num_epochs if not explicitly set
    max_steps = config.max_steps
    if max_steps is None and hasattr(dataset, "num_examples"):
        dataset_steps_per_gen = num_prompts_per_batch * dataset.data_world_size
        gen_batches_per_epoch = (dataset.num_examples + dataset_steps_per_gen - 1) // dataset_steps_per_gen
        steps_per_gen_batch = (config.data.batch_size // mini_batch_size) * config.generation.num_iterations
        max_steps = gen_batches_per_epoch * steps_per_gen_batch * config.num_epochs
        logger.info(f"Training for {config.num_epochs} epochs ({max_steps} optimizer steps, {dataset.num_examples} examples)")

    logger.info(f"Setting up {config.scheduler.type} scheduler with {max_steps} steps")
    scheduler = setup_scheduler(optimizer, config.scheduler, max_steps, config.optim.lr)

    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")

    api_key = os.environ.get("VLLM_API_KEY", "dummy")
    base_url = config.client.base_url[0]
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=config.client.timeout)

    # Loss config shortcuts
    distillation_topk = config.loss.distillation_topk if config.loss.full_logit_distillation else None
    distill_is_clip = config.loss.is_clip
    rollout_is = config.loss.rollout_is
    rollout_is_threshold = config.loss.rollout_is_threshold
    teacher_regularization = config.ref_model.regularization
    # Match SDPO behavior: if IS is configured, apply it regardless of step count per generation batch.
    needs_distill_is = distill_is_clip is not None
    needs_rollout_is = rollout_is is not None
    needs_old_log_probs = needs_distill_is or needs_rollout_is

    logger.info(f"Starting training loop (max_steps={max_steps or 'infinite'})")
    max_memory = torch.cuda.mem_get_info()[1] / 1024**3
    train_start_time = time.perf_counter()
    snapshot_hours = sorted(config.snapshot_hours)
    next_snapshot_idx = 0

    buffer = []
    buffer_idx = 0
    last_microbatch_signature = None

    while True:
        is_last_step = max_steps is not None and progress.step == max_steps

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

        if max_steps is not None and progress.step >= max_steps:
            break

        # === Generation phase ===
        if not buffer or buffer_idx >= len(buffer):
            torch.cuda.reset_peak_memory_stats()
            step_start_time = time.perf_counter()

            prompt_batch = next(dataset_iter)
            prompts = prompt_batch["prompts"]
            answers = prompt_batch["answers"]
            systems = prompt_batch["systems"]
            kinds = prompt_batch["kinds"]
            num_completions = config.generation.num_completions

            if world.is_master:
                logger.info(f"Generating {len(prompts)}x{num_completions} completions...")
                completions = asyncio.get_event_loop().run_until_complete(
                    generate_completions(
                        client=client,
                        prompts=prompts,
                        system_messages=systems,
                        model_name=config.model.name,
                        max_tokens=config.generation.max_completion_length,
                        temperature=config.generation.temperature,
                        top_p=config.generation.top_p,
                        top_k=config.generation.top_k,
                        num_completions=num_completions,
                    )
                )

                # Score completions
                scores = []
                for i, completion in enumerate(completions):
                    prompt_idx = i // num_completions
                    scores.append(score_completion(completion, answers[prompt_idx], kinds[prompt_idx]))

                # Build teacher prompts with successful demonstrations
                teacher_prompts, sd_mask = build_teacher_prompts(
                    prompts=prompts,
                    completions=completions,
                    scores=scores,
                    num_completions=num_completions,
                    reprompt_cfg=config.reprompt,
                )

                # Expand student prompts to match (each prompt repeated num_completions times)
                student_prompts = [p for p in prompts for _ in range(num_completions)]
                student_systems = [s for s in systems for _ in range(num_completions)]
                teacher_systems = [s for s in systems for _ in range(num_completions)]

                num_success = sum(1 for s in scores if s["score"] >= config.reprompt.success_threshold)
                num_reprompted = sum(sd_mask)
                avg_score = sum(s["score"] for s in scores) / len(scores)
                avg_comp_len = sum(len(c) for c in completions) / len(completions)
                gen_metrics = {
                    "gen/success_rate": num_success / len(scores),
                    "gen/reprompt_fraction": num_reprompted / len(sd_mask),
                    "gen/avg_score": avg_score,
                    "gen/avg_completion_length": avg_comp_len,
                }
                gen_data = (
                    completions,
                    student_prompts,
                    student_systems,
                    teacher_prompts,
                    teacher_systems,
                    sd_mask,
                    gen_metrics,
                )
                logger.info(
                    f"Scored: {num_success}/{len(scores)} successful, "
                    f"{num_reprompted}/{len(sd_mask)} reprompted"
                )
            else:
                gen_data = None

            # Broadcast generation data to all ranks
            if world.world_size > 1:
                if world.is_master:
                    data = pickle.dumps(gen_data)
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
                    gen_data = pickle.loads(data_tensor.cpu().numpy().tobytes())

            completions, student_prompts, student_systems, teacher_prompts, teacher_systems, sd_mask, gen_metrics = gen_data

            gen_time = time.perf_counter() - step_start_time
            gen_metrics["gen/time"] = gen_time
            monitor.log(gen_metrics, step=progress.step)
            logger.debug(f"Generation took {gen_time:.2f}s, avg completion length: {gen_metrics['gen/avg_completion_length']:.0f} chars")

            train_batch = prepare_sdft_batch(
                student_prompts=student_prompts,
                teacher_prompts=teacher_prompts,
                completions=completions,
                self_distillation_mask=sd_mask,
                tokenizer=tokenizer,
                max_prompt_length=config.generation.max_prompt_length,
                max_completion_length=config.generation.max_completion_length,
                max_reprompt_length=config.reprompt.max_reprompt_length,
                reprompt_truncation=config.reprompt.reprompt_truncation,
                student_systems=student_systems,
                teacher_systems=teacher_systems,
            )

            # Split into mini-batches, then micro-batches within each
            buffer = []
            bs = config.data.micro_batch_size
            for mb_idx in range(0, config.data.batch_size, bs):
                mb = {k: v[mb_idx : mb_idx + bs] for k, v in train_batch.items()}
                buffer.append(mb)

            student_seq_len = train_batch["student_input_ids"].shape[1]
            teacher_seq_len = train_batch["teacher_input_ids"].shape[1]
            max_seq_len = max(student_seq_len, teacher_seq_len)
            micro_batch_count = len(buffer)
            completion_tokens = train_batch["completion_mask"].sum(dim=-1)
            max_completion_tokens = int(completion_tokens.max().item()) if completion_tokens.numel() > 0 else 0
            avg_completion_tokens = completion_tokens.float().mean().item() if completion_tokens.numel() > 0 else 0.0

            signature = (bs, micro_batch_count, student_seq_len, teacher_seq_len, max_completion_tokens)
            if signature != last_microbatch_signature:
                last_microbatch_signature = signature
                vocab_size = model.lm_head.weight.shape[0]
                tokens_per_micro_batch = bs * max_seq_len
                logits_bf16_gib = _estimate_logits_memory_gib(tokens_per_micro_batch, vocab_size, bytes_per_elem=2)
                logits_fp32_gib = _estimate_logits_memory_gib(tokens_per_micro_batch, vocab_size, bytes_per_elem=4)

                logger.info(
                    "Microbatching summary: "
                    f"micro_batch_size={bs}, micro_batches_per_gen={micro_batch_count}, "
                    f"grad_accum_steps={grad_accum_steps}, student_seq_len={student_seq_len}, "
                    f"teacher_seq_len={teacher_seq_len}, completion_tokens(avg/max)={avg_completion_tokens:.1f}/{max_completion_tokens}"
                )
                logger.info(
                    "Microbatching memory estimate (single [B,S,V] tensor): "
                    f"tokens_per_micro_batch={tokens_per_micro_batch}, vocab={vocab_size}, "
                    f"bf16={logits_bf16_gib:.1f} GiB, fp32={logits_fp32_gib:.1f} GiB"
                )
                if logits_fp32_gib > max_memory * 0.7:
                    logger.warning(
                        "Estimated fp32 logits tensor is >=70% of device memory. "
                        f"Consider lowering micro_batch_size (current {bs})."
                    )

            # Compute rollout-time logprobs before optimizer updates for IS correction.
            if needs_old_log_probs:
                logger.debug("Computing rollout-time logprobs for IS correction")
                with torch.no_grad():
                    for mb in buffer:
                        ids = mb["student_input_ids"].to("cuda")
                        pos = mb["student_position_ids"].to("cuda")
                        hidden = forward_hidden(model, ids, pos)
                        token_lps = _compute_token_log_probs_from_hidden(
                            hidden=hidden,
                            input_ids=ids,
                            lm_weight=model.lm_head.weight,
                            chunk_size=config.loss.distillation_chunk_size,
                        )
                        mb["old_token_log_probs"] = token_lps.cpu()
                        del hidden

            # Repeat buffer for multi-iteration (off-policy) training
            if config.generation.num_iterations > 1:
                buffer = buffer * config.generation.num_iterations

            buffer_idx = 0

            logger.debug(f"Batch shapes: student={train_batch['student_input_ids'].shape}, teacher={train_batch['teacher_input_ids'].shape}, completion_tokens={train_batch['completion_mask'].sum().item()}")

            broadcast_weights_to_inference(model, config.output_dir, progress.step, world)

        # === Training phase ===
        train_step_start = time.perf_counter()
        batch_loss = torch.tensor(0.0, device="cuda")
        batch_kl = torch.tensor(0.0, device="cuda")
        batch_is_mean = torch.tensor(0.0, device="cuda")
        batch_is_max = torch.tensor(0.0, device="cuda")
        micro_steps_available = len(buffer) - buffer_idx
        micro_steps_this_step = min(grad_accum_steps, max(micro_steps_available, 0))

        for micro_step in range(grad_accum_steps):
            if buffer_idx >= len(buffer):
                break
            micro_step_idx = micro_step + 1
            micro_batch = buffer[buffer_idx]
            buffer_idx += 1

            student_input_ids = micro_batch["student_input_ids"].to("cuda")
            student_position_ids = micro_batch["student_position_ids"].to("cuda")
            teacher_input_ids = micro_batch["teacher_input_ids"].to("cuda")
            teacher_position_ids = micro_batch["teacher_position_ids"].to("cuda")
            completion_mask = micro_batch["completion_mask"].to("cuda")
            teacher_completion_mask = micro_batch["teacher_completion_mask"].to("cuda")

            active_teacher = teacher_model if teacher_model is not None else model
            if teacher_regularization == "trust-region" and teacher_model is None:
                raise RuntimeError("trust-region teacher requires a separate reference model")
            sd_mask = micro_batch["self_distillation_mask"].to("cuda")
            student_comp_token_mask = _completion_token_mask(completion_mask)
            teacher_comp_token_mask = _completion_token_mask(teacher_completion_mask)

            if not torch.equal(student_comp_token_mask, teacher_comp_token_mask):
                raise RuntimeError("Student/teacher completion token masks are misaligned")

            aligned_mask = student_comp_token_mask & sd_mask.unsqueeze(1).bool()

            if config.loss.fused_distillation:
                if teacher_regularization == "trust-region":
                    raise RuntimeError("trust-region teacher requires fused_distillation to be disabled")
                # Fused path: operate on hidden states, never materialize [N, V]
                with maybe_activation_offloading(config.model.ac_offloading):
                    student_hidden = forward_hidden(model, student_input_ids, student_position_ids)
                with torch.no_grad():
                    teacher_hidden = forward_hidden(active_teacher, teacher_input_ids, teacher_position_ids)

                s_flat = student_hidden[completion_mask]
                t_flat = teacher_hidden[teacher_completion_mask]
                lm_weight = model.lm_head.weight

                student_topk_lp, teacher_topk_lp, _ = fused_distill_topk(
                    s_flat, t_flat, lm_weight,
                    K=distillation_topk,
                    chunk_size=config.loss.distillation_chunk_size,
                )

                student_topk_lp = _scatter_flat_by_mask(student_topk_lp, student_comp_token_mask)
                teacher_topk_lp = _scatter_flat_by_mask(teacher_topk_lp, student_comp_token_mask)

                if config.loss.distillation_add_tail:
                    student_distill_lp = add_tail(student_topk_lp)
                    teacher_distill_lp = add_tail(teacher_topk_lp)
                else:
                    student_distill_lp = renorm_topk_log_probs(student_topk_lp)
                    teacher_distill_lp = renorm_topk_log_probs(teacher_topk_lp)

                # IS correction from hidden states to avoid [B,S,V] logits materialization.
                is_ratio = None
                rollout_is_weights = None
                if needs_old_log_probs:
                    with torch.no_grad():
                        student_token_lp = _compute_token_log_probs_from_hidden(
                            hidden=student_hidden.detach(),
                            input_ids=student_input_ids,
                            lm_weight=lm_weight,
                            chunk_size=config.loss.distillation_chunk_size,
                        )
                    student_token_lp_comp = _scatter_flat_by_mask(student_token_lp[completion_mask], student_comp_token_mask)
                    old_token_lp_comp = micro_batch["old_token_log_probs"].to("cuda")
                    old_token_lp_comp = _scatter_flat_by_mask(old_token_lp_comp[completion_mask], student_comp_token_mask)

                    negative_approx_kl = (student_token_lp_comp - old_token_lp_comp).detach()
                    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                    if needs_distill_is:
                        is_ratio = torch.exp(negative_approx_kl).clamp(max=distill_is_clip)
                        batch_is_mean += (
                            (is_ratio * aligned_mask).sum().detach() / aligned_mask.sum().clamp(min=1) / grad_accum_steps
                        )
                        batch_is_max = torch.max(batch_is_max, is_ratio.max().detach())
                    if needs_rollout_is:
                        rollout_is_weights = _compute_rollout_is_weights(
                            log_ratio=negative_approx_kl,
                            mask=aligned_mask,
                            rollout_is=rollout_is,
                            threshold=rollout_is_threshold,
                        )

                loss, metrics = sdft_kl_loss(
                    student_distill_lp,
                    teacher_distill_lp,
                    aligned_mask,
                    alpha=config.loss.alpha,
                    is_ratio=is_ratio,
                    rollout_is_weights=rollout_is_weights,
                )

                batch_loss += loss.detach() / grad_accum_steps
                batch_kl += metrics["kl_divergence"] / grad_accum_steps

                del student_hidden, teacher_hidden
                del student_distill_lp, teacher_distill_lp

                (loss / grad_accum_steps).backward()

            else:
                # Standard path: full [N, V] logit tensors
                with maybe_activation_offloading(config.model.ac_offloading):
                    student_out = forward(
                        model,
                        student_input_ids,
                        student_position_ids,
                        cast_output_to_float=False,
                    )
                student_logits = student_out["logits"]

                if teacher_regularization == "trust-region":
                    with torch.no_grad():
                        teacher_out = forward(
                            teacher_model,
                            teacher_input_ids,
                            teacher_position_ids,
                            cast_output_to_float=False,
                        )
                        teacher_logits = torch.lerp(
                            teacher_out["logits"],
                            student_logits.detach(),
                            config.ref_model.update_rate,
                        )
                else:
                    with torch.no_grad():
                        teacher_out = forward(
                            active_teacher,
                            teacher_input_ids,
                            teacher_position_ids,
                            cast_output_to_float=False,
                        )
                        teacher_logits = teacher_out["logits"]

                student_comp_logits = _extract_completion_logits(student_logits, completion_mask)
                teacher_comp_logits = _extract_completion_logits(teacher_logits, teacher_completion_mask)

                student_comp_log_probs = student_comp_logits.log_softmax(dim=-1)
                with torch.no_grad():
                    teacher_comp_log_probs = teacher_comp_logits.log_softmax(dim=-1)

                if distillation_topk is not None:
                    student_topk_lp, topk_idx = student_comp_log_probs.topk(distillation_topk, dim=-1)
                    with torch.no_grad():
                        teacher_topk_lp = teacher_comp_log_probs.gather(-1, topk_idx)

                    if config.loss.distillation_add_tail:
                        student_distill_lp = add_tail(student_topk_lp)
                        teacher_distill_lp = add_tail(teacher_topk_lp)
                    else:
                        student_distill_lp = renorm_topk_log_probs(student_topk_lp)
                        teacher_distill_lp = renorm_topk_log_probs(teacher_topk_lp)
                else:
                    student_distill_lp = student_comp_log_probs
                    teacher_distill_lp = teacher_comp_log_probs

                is_ratio = None
                rollout_is_weights = None
                if needs_old_log_probs:
                    student_token_lp = _compute_token_log_probs(student_logits, student_input_ids)
                    student_token_lp_comp = _extract_completion_values(student_token_lp, completion_mask)
                    old_token_lp_comp = micro_batch["old_token_log_probs"].to("cuda")
                    old_token_lp_comp = _extract_completion_values(old_token_lp_comp, completion_mask)

                    negative_approx_kl = (student_token_lp_comp - old_token_lp_comp).detach()
                    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                    if needs_distill_is:
                        is_ratio = torch.exp(negative_approx_kl).clamp(max=distill_is_clip)
                        batch_is_mean += (
                            (is_ratio * aligned_mask).sum().detach() / aligned_mask.sum().clamp(min=1) / grad_accum_steps
                        )
                        batch_is_max = torch.max(batch_is_max, is_ratio.max().detach())
                    if needs_rollout_is:
                        rollout_is_weights = _compute_rollout_is_weights(
                            log_ratio=negative_approx_kl,
                            mask=aligned_mask,
                            rollout_is=rollout_is,
                            threshold=rollout_is_threshold,
                        )

                loss, metrics = sdft_kl_loss(
                    student_distill_lp,
                    teacher_distill_lp,
                    aligned_mask,
                    alpha=config.loss.alpha,
                    is_ratio=is_ratio,
                    rollout_is_weights=rollout_is_weights,
                )

                batch_loss += loss.detach() / grad_accum_steps
                batch_kl += metrics["kl_divergence"] / grad_accum_steps

                del student_logits, teacher_logits, student_comp_logits, teacher_comp_logits
                del student_comp_log_probs, teacher_comp_log_probs, student_distill_lp, teacher_distill_lp

                (loss / grad_accum_steps).backward()

            logger.info(
                "Microbatch done: "
                f"optimizer_step={progress.step}, "
                f"micro_step={micro_step_idx}/{micro_steps_this_step}, "
                f"buffer={buffer_idx}/{len(buffer)}"
            )

        # Optimizer step
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm, ep_enabled=False)
        if grad_norm.device.type == "cpu":
            grad_norm = grad_norm.to("cuda")

        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # EMA update after every optimizer step
        if teacher_model is not None and teacher_regularization == "ema":
            ema_update(teacher_model, model, config.ref_model.update_rate)
            logger.debug(f"EMA update at step {progress.step + 1} (rate={config.ref_model.update_rate})")

        # Synchronize metrics
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_kl, op=dist.ReduceOp.AVG)

        step_time = time.perf_counter() - train_step_start
        elapsed = time.perf_counter() - train_start_time
        elapsed_h = elapsed / 3600
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3

        step_message = (
            f"Step {progress.step} | "
            f"Time: {step_time:.2f}s | "
            f"Elapsed: {elapsed_h:.2f}h | "
            f"KL Loss: {batch_loss.item():.4f} | "
            f"KL Div: {batch_kl.item():.4f} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Peak Mem: {peak_memory:.1f}/{max_memory:.1f} GiB"
        )
        logger.success(step_message)

        log_dict = {
            "loss/kl": batch_loss.item(),
            "loss/kl_divergence": batch_kl.item(),
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "perf/peak_memory": peak_memory,
            "time/step": step_time,
            "time/elapsed_hours": elapsed_h,
            "step": progress.step,
        }
        if needs_distill_is:
            log_dict["is/weight_mean"] = batch_is_mean.item()
            log_dict["is/weight_max"] = batch_is_max.item()
        monitor.log(log_dict, step=progress.step)

        # Time-based snapshots
        if (
            next_snapshot_idx < len(snapshot_hours)
            and elapsed_h >= snapshot_hours[next_snapshot_idx]
            and weight_ckpt_manager is not None
        ):
            hour_tag = snapshot_hours[next_snapshot_idx]
            logger.info(f"Saving {hour_tag}h snapshot at step {progress.step} ({elapsed_h:.2f}h elapsed)")
            weight_ckpt_manager.save(progress.step, model, tokenizer)
            next_snapshot_idx += 1

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


def main():
    train(parse_argv(SDFTTrainerConfig))


if __name__ == "__main__":
    main()
