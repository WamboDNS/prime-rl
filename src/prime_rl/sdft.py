import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread
from typing import Annotated

import tomli_w
from pydantic import Field, model_validator

from prime_rl.inference.config import InferenceConfig
from prime_rl.trainer.sdft.config import SDFTTrainerConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pydantic_config import BaseSettings, parse_argv
from prime_rl.utils.utils import get_free_port, get_log_dir


class SDFTConfig(BaseSettings):
    """Top-level configuration for an SDFT training run."""

    trainer: SDFTTrainerConfig
    inference: Annotated[
        InferenceConfig | None,
        Field(description="Inference server config. If None, assumes an external server is running."),
    ] = None

    inference_gpu_ids: Annotated[list[int], Field(description="GPU IDs for the inference server.")] = [0]
    trainer_gpu_ids: Annotated[list[int], Field(description="GPU IDs for the trainer.")] = [1]

    output_dir: Annotated[
        Path,
        Field(description="Directory for outputs (checkpoints, logs)."),
    ] = Path("outputs")

    clean: Annotated[
        bool,
        Field(description="Whether to clean output directories before starting."),
    ] = True

    @model_validator(mode="after")
    def auto_setup_output_dir(self):
        if self.output_dir is not None:
            self.trainer.output_dir = self.output_dir
        return self

    @model_validator(mode="after")
    def auto_setup_model(self):
        if self.inference is not None:
            self.inference.model.name = self.trainer.model.name
        return self

    @model_validator(mode="after")
    def auto_setup_inference_dp(self):
        if self.inference and len(self.inference_gpu_ids) != self.inference.parallel.dp * self.inference.parallel.tp:
            assert len(self.inference_gpu_ids) % self.inference.parallel.tp == 0, (
                "Number of inference GPUs must be divisible by the tensor parallel size"
            )
            self.inference.parallel.dp = len(self.inference_gpu_ids) // self.inference.parallel.tp
        return self

    @model_validator(mode="after")
    def auto_setup_client_url(self):
        if self.inference is not None:
            host = self.inference.server.host or "localhost"
            port = self.inference.server.port
            self.trainer.client.base_url = [f"http://{host}:{port}/v1"]
        return self


def monitor_process(process: Popen, stop_event: Event, error_queue: list, process_name: str):
    """Monitor a subprocess and signal errors via shared queue."""
    process.wait()
    if process.returncode != 0:
        err_msg = f"{process_name} failed with exit code {process.returncode}"
        error_queue.append(RuntimeError(err_msg))
    stop_event.set()


def cleanup_processes(processes: list[Popen]):
    for process in processes:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=60)


def sdft(config: SDFTConfig):
    logger = setup_logger("info", log_file=config.output_dir / "logs" / "sdft.log")
    logger.info("Starting SDFT run")

    log_dir = get_log_dir(config.output_dir)

    if config.clean:
        logger.info("Cleaning output directories")
        shutil.rmtree(log_dir, ignore_errors=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Write resolved subconfigs to disk
    config_dir = Path(".pydantic_config") / uuid.uuid4().hex
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / "trainer.toml", "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        with open(config_dir / "inference.toml", "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    start_command = sys.argv

    try:
        # Start inference server
        if config.inference:
            inference_cmd = ["uv", "run", "inference", "@", (config_dir / "inference.toml").as_posix()]
            logger.info(f"Starting inference server on GPU(s) {config.inference_gpu_ids}")
            with open(log_dir / "inference.stdout", "w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.inference_gpu_ids))},
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            stop_event = Event()
            stop_events["inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, "inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        else:
            logger.warning("No inference config specified. Is your inference server running?")

        # Start trainer
        trainer_cmd = [
            "uv",
            "run",
            "env",
            "PYTHONUNBUFFERED=1",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "torchrun",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            f"--log-dir={config.output_dir / 'torchrun'}",
            "--local-ranks-filter=0",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(config.trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.sdft.train",
            "@",
            (config_dir / "trainer.toml").as_posix(),
        ]
        logger.info(f"Starting SDFT trainer on GPU(s) {config.trainer_gpu_ids}")
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, config.trainer_gpu_ids)),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "WANDB_PROGRAM": "uv run sdft",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(trainer_process, stop_event, error_queue, "trainer"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        logger.success("Startup complete. Showing trainer logs...")
        tail_process = Popen(["tail", "-F", log_dir / "trainer.stdout"])
        processes.append(tail_process)

        # Monitor until trainer finishes
        while not stop_events["trainer"].is_set():
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                cleanup_processes(processes)
                sys.exit(1)
            time.sleep(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("SDFT training finished!")
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt, terminating all processes...")
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        cleanup_processes(processes)
        raise


def main():
    sdft(parse_argv(SDFTConfig))


if __name__ == "__main__":
    main()
