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
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.sdft.config import SDFTRepromptConfig, SDFTTrainerConfig
from prime_rl.trainer.sdft.data import (
    prepare_sdft_batch,
    setup_sdft_dataset,
)
from prime_rl.trainer.sdft.fused_distill import fused_distill_topk
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
        for _ in range(num_completions):
            task = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={"top_k": top_k},
            )
            tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content or "" for r in responses]


def _remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def build_teacher_prompts(
    prompts: list[str],
    completions: list[str],
    scores: list[dict],
    systems: list[str | None],
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
        has_feedback = feedback is not None

        if has_solution or has_feedback:
            solution_section = ""
            if has_solution:
                solution_text = completions[solution_idxs[0]]
                if reprompt_cfg.remove_thinking:
                    solution_text = _remove_thinking(solution_text)
                solution_section = reprompt_cfg.solution_template.format(
                    successful_previous_attempt=solution_text,
                )

            feedback_section = ""
            if has_feedback:
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


def forward_hidden(model: torch.nn.Module, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Get hidden states from the model backbone without going through lm_head."""
    outputs = model.model(input_ids=input_ids, position_ids=position_ids)
    return outputs.last_hidden_state


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

    # EMA teacher model
    teacher_model = None
    if config.ref_model.enabled:
        logger.info("Initializing EMA teacher model")
        teacher_model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
        for param in teacher_model.parameters():
            param.requires_grad = False

    logger.info(f"Initializing optimizer ({config.optim})")
    optimizer = setup_optimizer(config.optim, list(model.named_parameters()), parallel_dims)

    scheduler_steps = config.max_steps
    logger.info(f"Setting up {config.scheduler.type} scheduler with {scheduler_steps} steps")
    scheduler = setup_scheduler(optimizer, config.scheduler, scheduler_steps, config.optim.lr)

    num_prompts_per_batch = config.data.batch_size // config.generation.num_completions
    logger.info(f"Initializing dataset ({config.data})")
    dataset = setup_sdft_dataset(config.data, num_prompts_per_batch)
    dataset_iter = iter(dataset)

    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")

    api_key = os.environ.get("VLLM_API_KEY", "dummy")
    base_url = config.client.base_url[0]
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=config.client.timeout)

    # Loss config shortcuts
    distillation_topk = config.loss.distillation_topk if config.loss.full_logit_distillation else None
    is_clip = config.loss.is_clip
    # IS correction is needed when the model changes between processing samples in the same generation batch:
    # either multiple optimizer steps per batch (mini_batch_size < batch_size) or multiple iterations.
    single_step_per_batch = (config.generation.num_iterations == 1 and mini_batch_size == config.data.batch_size)
    needs_is = is_clip is not None and not single_step_per_batch

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
                    systems=systems,
                    num_completions=num_completions,
                    reprompt_cfg=config.reprompt,
                )

                # Expand student prompts to match (each prompt repeated num_completions times)
                student_prompts = [p for p in prompts for _ in range(num_completions)]

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
                gen_data = (completions, student_prompts, teacher_prompts, sd_mask, gen_metrics)
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

            completions, student_prompts, teacher_prompts, sd_mask, gen_metrics = gen_data

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
            )

            # Split into mini-batches, then micro-batches within each
            buffer = []
            bs = config.data.micro_batch_size
            for mb_idx in range(0, config.data.batch_size, bs):
                mb = {k: v[mb_idx : mb_idx + bs] for k, v in train_batch.items()}
                buffer.append(mb)

            # Compute old logprobs for IS correction before any optimizer steps
            if needs_is:
                logger.debug("Computing old logprobs for IS correction")
                with torch.no_grad():
                    for mb in buffer:
                        ids = mb["student_input_ids"].to("cuda")
                        pos = mb["student_position_ids"].to("cuda")
                        out = forward(model, ids, pos)
                        token_lps = _compute_token_log_probs(out["logits"], ids)
                        mb["old_token_log_probs"] = token_lps.cpu()

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

        for micro_step in range(grad_accum_steps):
            if buffer_idx >= len(buffer):
                break
            micro_batch = buffer[buffer_idx]
            buffer_idx += 1

            student_input_ids = micro_batch["student_input_ids"].to("cuda")
            student_position_ids = micro_batch["student_position_ids"].to("cuda")
            teacher_input_ids = micro_batch["teacher_input_ids"].to("cuda")
            teacher_position_ids = micro_batch["teacher_position_ids"].to("cuda")
            completion_mask = micro_batch["completion_mask"].to("cuda")
            teacher_completion_mask = micro_batch["teacher_completion_mask"].to("cuda")

            active_teacher = teacher_model if teacher_model is not None else model
            sd_mask = micro_batch["self_distillation_mask"].to("cuda")

            if config.loss.fused_distillation:
                # Fused path: operate on hidden states, never materialize [N, V]
                with maybe_activation_offloading(config.model.ac_offloading):
                    student_hidden = forward_hidden(model, student_input_ids, student_position_ids)
                with torch.no_grad():
                    teacher_hidden = forward_hidden(active_teacher, teacher_input_ids, teacher_position_ids)

                student_comp_hidden = _extract_completion_logits(student_hidden, completion_mask)
                teacher_comp_hidden = _extract_completion_logits(teacher_hidden, teacher_completion_mask)

                B = student_comp_hidden.shape[0]
                comp_len = student_comp_hidden.shape[1]
                H = student_comp_hidden.shape[2]

                s_flat = student_comp_hidden.reshape(-1, H)
                t_flat = teacher_comp_hidden.reshape(-1, H)
                lm_weight = model.lm_head.weight

                student_topk_lp, teacher_topk_lp, _ = fused_distill_topk(
                    s_flat, t_flat, lm_weight,
                    K=distillation_topk,
                    chunk_size=config.loss.distillation_chunk_size,
                )

                student_topk_lp = student_topk_lp.reshape(B, comp_len, distillation_topk)
                teacher_topk_lp = teacher_topk_lp.reshape(B, comp_len, distillation_topk)

                if config.loss.distillation_add_tail:
                    student_distill_lp = add_tail(student_topk_lp)
                    teacher_distill_lp = add_tail(teacher_topk_lp)
                else:
                    student_distill_lp = renorm_topk_log_probs(student_topk_lp)
                    teacher_distill_lp = renorm_topk_log_probs(teacher_topk_lp)

                aligned_mask = torch.ones(B, comp_len, dtype=torch.bool, device="cuda")
                aligned_mask = aligned_mask * sd_mask.unsqueeze(1).bool()

                # IS correction still needs full forward for token-level log probs
                is_ratio = None
                if needs_is:
                    with torch.no_grad():
                        student_out = forward(model, student_input_ids, student_position_ids)
                        student_logits = student_out["logits"]
                    student_token_lp = _compute_token_log_probs(student_logits, student_input_ids)
                    student_token_lp_comp = _extract_completion_values(student_token_lp, completion_mask)
                    old_token_lp_comp = micro_batch["old_token_log_probs"].to("cuda")
                    old_token_lp_comp = _extract_completion_values(old_token_lp_comp, completion_mask)

                    negative_approx_kl = (student_token_lp_comp - old_token_lp_comp).detach()
                    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                    is_ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)

                    batch_is_mean += (is_ratio * aligned_mask).sum().detach() / aligned_mask.sum().clamp(min=1) / grad_accum_steps
                    batch_is_max = torch.max(batch_is_max, is_ratio.max().detach())
                    del student_logits

                loss, metrics = sdft_kl_loss(
                    student_distill_lp,
                    teacher_distill_lp,
                    aligned_mask,
                    alpha=config.loss.alpha,
                    is_ratio=is_ratio,
                )

                batch_loss += loss.detach() / grad_accum_steps
                batch_kl += metrics["kl_divergence"] / grad_accum_steps

                del student_hidden, teacher_hidden, student_comp_hidden, teacher_comp_hidden
                del student_distill_lp, teacher_distill_lp

                (loss / grad_accum_steps).backward()

            else:
                # Standard path: full [N, V] logit tensors
                with maybe_activation_offloading(config.model.ac_offloading):
                    student_out = forward(model, student_input_ids, student_position_ids)
                student_logits = student_out["logits"]

                with torch.no_grad():
                    teacher_out = forward(active_teacher, teacher_input_ids, teacher_position_ids)
                    teacher_logits = teacher_out["logits"]

                student_comp_logits = _extract_completion_logits(student_logits, completion_mask)
                teacher_comp_logits = _extract_completion_logits(teacher_logits, teacher_completion_mask)

                comp_len = student_comp_logits.shape[1]
                aligned_mask = torch.ones(student_comp_logits.shape[0], comp_len, dtype=torch.bool, device="cuda")
                aligned_mask = aligned_mask * sd_mask.unsqueeze(1).bool()

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
                if needs_is:
                    student_token_lp = _compute_token_log_probs(student_logits, student_input_ids)
                    student_token_lp_comp = _extract_completion_values(student_token_lp, completion_mask)
                    old_token_lp_comp = micro_batch["old_token_log_probs"].to("cuda")
                    old_token_lp_comp = _extract_completion_values(old_token_lp_comp, completion_mask)

                    negative_approx_kl = (student_token_lp_comp - old_token_lp_comp).detach()
                    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
                    is_ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)

                    batch_is_mean += (is_ratio * aligned_mask).sum().detach() / aligned_mask.sum().clamp(min=1) / grad_accum_steps
                    batch_is_max = torch.max(batch_is_max, is_ratio.max().detach())

                loss, metrics = sdft_kl_loss(
                    student_distill_lp,
                    teacher_distill_lp,
                    aligned_mask,
                    alpha=config.loss.alpha,
                    is_ratio=is_ratio,
                )

                batch_loss += loss.detach() / grad_accum_steps
                batch_kl += metrics["kl_divergence"] / grad_accum_steps

                del student_logits, teacher_logits, student_comp_logits, teacher_comp_logits
                del student_comp_log_probs, teacher_comp_log_probs, student_distill_lp, teacher_distill_lp

                (loss / grad_accum_steps).backward()

        # Optimizer step
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.optim.max_norm, ep_enabled=False)
        if grad_norm.device.type == "cpu":
            grad_norm = grad_norm.to("cuda")

        optimizer.step()
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # EMA update after every optimizer step
        if teacher_model is not None:
            ema_update(teacher_model, model, config.ref_model.update_rate)
            logger.debug(f"EMA update at step {progress.step + 1} (rate={config.ref_model.update_rate})")

        # Synchronize metrics
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(batch_kl, op=dist.ReduceOp.AVG)

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

        log_dict = {
            "loss/kl": batch_loss.item(),
            "loss/kl_divergence": batch_kl.item(),
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "perf/peak_memory": peak_memory,
            "time/step": step_time,
            "step": progress.step,
        }
        if needs_is:
            log_dict["is/weight_mean"] = batch_is_mean.item()
            log_dict["is/weight_max"] = batch_is_max.item()
        monitor.log(log_dict, step=progress.step)

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
