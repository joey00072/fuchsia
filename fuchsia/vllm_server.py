# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# derived from https://github.com/huggingface/trl/blob/main/trl/scripts/vllm_serve.py
# from original pr of binary-husky (https://github.com/binary-husky) pr https://github.com/huggingface/trl/pull/3094

# Standard library imports
import argparse
import asyncio
import ctypes
from contextlib import contextmanager
import logging
import os
import threading
import time
from typing import Optional, Sequence, Callable

# Third party imports
import torch
import torch.distributed as dist
import uvicorn
from datasets import Dataset
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

# Check CUDA availability
try:
    ctypes.CDLL("libcuda.so.1")
    libcuda_available = True
except OSError:
    libcuda_available = False

# VLLM imports if CUDA is available
if libcuda_available:
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

# Local imports
from fuchsia.envs import Rollout, RolloutSample, Environment, SingleTurnEnvironment
from fuchsia.rollout_queue import create_rollout_queue
from fuchsia.config import FuchsiaConfig
from fuchsia.utils import get_ip_addresses

# Configure logging
logger = logging.getLogger(__name__)

# Configure multiprocessing for CUDA
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"



class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker extension uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    pynccl_comm = None
    client_rank = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized")

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(
        self, name: str, dtype: torch.dtype, shape: Sequence[int]
    ) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """

        if self.pynccl_comm is None:
            print("Communicator not initialized")
            return
            # raise RuntimeError("Communicator not initialized")

        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(
            weight, src=self.client_rank, stream=torch.cuda.current_stream()
        )
        self.pynccl_comm.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None

# API Models
class GenerateRequest(BaseModel):
    prompts: list[str]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    guided_decoding_regex: Optional[str] = None


class GenerateResponse(BaseModel):
    completion_ids: list[list[int]]


class InitCommunicatorRequest(BaseModel):
    host: str
    port: int
    world_size: int


class UpdateWeightsRequest(BaseModel):
    name: str
    dtype: str
    shape: list[int]


class VLLMServer:
    def __init__(
        self, 
        config: FuchsiaConfig,
        dataset: Optional[Dataset] = None,
        reward_functions: Optional[list[Callable]] = None,
        pre_fill_buffer: bool = True,
        environment: Environment = None,
        stop: Optional[list[str]] = None,
    ):
        self.config = config
        logger.info(config)
        self.stop = stop
        self._prefill_buffer_on_startup = pre_fill_buffer and dataset is not None

        if not os.environ.get('VLLM_ATTENTION_BACKEND'):
            # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLEX_ATTENTION'
            logger.info("Set VLLM_ATTENTION_BACKEND to TRITON")
        
        kwargs = {}
        if config.vllm_kv_quantization:
            kwargs["kv_cache_dtype"] = "fp8"
            kwargs["calculate_kv_scales"] = True
        print(config)

        self.llm = LLM(
            model=config.model,
            quantization=config.quantization,
            revision=config.revision,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            enable_prefix_caching=config.enable_prefix_caching,
            max_model_len=config.max_model_len,
            enable_lora=config.single_gpu,
            enable_sleep_mode=True,  # Enable sleep mode for CUDA
            worker_extension_cls="fuchsia.vllm_server.WeightSyncWorkerExtension",
            **kwargs
        )

        # Data sampler specific initialization
        self.dataset = dataset
        self.is_data_sampler = dataset is not None
        self._lora_request_id = 1
        
        self.environment = environment or SingleTurnEnvironment(reward_functions=reward_functions)
        
        if self.is_data_sampler:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model)
            self.dataset_field = config.dataset_field
            self.buffer_size = config.buffer_size
            if config.generation_batch_size <= 0:
                raise ValueError(
                    f"generation_batch_size must be > 0, got {config.generation_batch_size}"
                )
            self.rollout_queue = create_rollout_queue(
                mode=config.sample_transfer_mode,
                queue_dir=config.sample_transfer_dir,
                clear_on_start=config.sample_transfer_clear_on_start,
            )
            self.sample_transfer_mode = self.rollout_queue.mode
            if self.rollout_queue.queue_dir is not None:
                logger.info(
                    "Data sampler transfer mode: filesystem (%s)",
                    self.rollout_queue.queue_dir,
                )
            else:
                logger.info("Data sampler transfer mode: api (in-memory queue)")
            self.dataset_iter = iter(self.dataset)
            self._epoch = 1
            self._generation_batch_size = config.generation_batch_size
            self._sampling_params = SamplingParams(
                n=config.vllm_n,
                repetition_penalty=config.vllm_repetition_penalty,
                temperature=config.vllm_temperature,
                top_p=config.vllm_top_p,
                top_k=config.vllm_top_k,
                min_p=config.vllm_min_p,
                logprobs=config.vllm_logprobs,
                max_tokens=config.vllm_max_tokens,
                stop=self.stop,
            )
            self.enable_lora = config.enable_lora
            self.lora_path = config.lora_path
            self._is_filling = False
            self._fill_claim_lock = threading.Lock()
            self._is_sleeping = False  # Track sleep state
            self._sleep_requested = False  # Track if sleep has been requested
            self._generation_lock = threading.Lock()  # Lock for generation operations
            self._last_fill_started_at: float | None = None
            self._last_fill_finished_at: float | None = None
            self._last_fill_heartbeat_at: float | None = None
            self._last_fill_duration_seconds: float | None = None
            self._last_fill_error: str | None = None
            self._last_fill_generated_groups: int = 0

        self.app = self._create_app()

    def _rollout_queue_size(self) -> int:
        if not self.is_data_sampler:
            return 0
        return self.rollout_queue.qsize()

    def _rollout_queue_push(self, items_with_rewards: list[dict]) -> None:
        self.rollout_queue.put_many(items_with_rewards)

    def _rollout_queue_pop(self):
        return self.rollout_queue.get()

    def _rollout_queue_clear(self) -> int:
        return self.rollout_queue.clear()

    def _sleep_block_reason(self) -> Optional[str]:
        if not self.is_data_sampler:
            return None
        if getattr(self, "_is_sleeping", False):
            return "LLM is sleeping"
        if getattr(self, "_sleep_requested", False):
            return "sleep requested"
        return None

    def _generation_blocked_by_sleep(self) -> bool:
        return bool(self.is_data_sampler and getattr(self, "_sleep_requested", False))

    def _current_generation_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.config.single_gpu and os.path.exists(self.lora_path):
            kwargs["lora_request"] = LoRARequest("grpo", self._lora_request_id, self.lora_path)
        return kwargs

    def _next_dataset_items(self, count: int) -> list[dict]:
        items: list[dict] = []
        for _ in range(count):
            try:
                item = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                self._epoch += 1
                try:
                    item = next(self.dataset_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Dataset iterator is empty; cannot fill rollout buffer"
                    ) from exc
            items.append(item)
        return items

    @contextmanager
    def _fill_claim(self, claim_acquired: bool):
        acquired_here = False
        if not claim_acquired:
            if not self._fill_claim_lock.acquire(blocking=False):
                yield False
                return
            acquired_here = True
        try:
            yield True
        finally:
            if claim_acquired or acquired_here:
                self._fill_claim_lock.release()

    def _start_fill_metrics(self) -> float:
        started_at = time.time()
        self._is_filling = True
        self._last_fill_started_at = started_at
        self._last_fill_heartbeat_at = started_at
        self._last_fill_finished_at = None
        self._last_fill_duration_seconds = None
        self._last_fill_error = None
        self._last_fill_generated_groups = 0
        return started_at

    def _finish_fill_metrics(self, started_at: float) -> None:
        self._is_filling = False
        finished_at = time.time()
        self._last_fill_finished_at = finished_at
        self._last_fill_duration_seconds = max(0.0, finished_at - started_at)

    def _environment_payload_for_items(
        self,
        items: list[dict],
        *,
        generation_kwargs: dict,
        max_items_per_generate_call: int,
    ) -> list[dict]:
        items_with_rewards: list[dict] = []
        for start_idx in range(0, len(items), max_items_per_generate_call):
            chunk_items = items[start_idx : start_idx + max_items_per_generate_call]
            rollouts: list[Rollout] = []
            for item in chunk_items:
                prompt_text = item[self.dataset_field]
                prompt_ids = self.tokenizer(
                    prompt_text, add_special_tokens=False
                )["input_ids"]
                rollouts.append(Rollout(prompt=prompt_text, prompt_ids=prompt_ids, item=item))

            # Keep heartbeat alive while vLLM generation is running.
            heartbeat_stop = threading.Event()
            heartbeat_thread = threading.Thread(
                target=self._fill_heartbeat_loop,
                args=(heartbeat_stop,),
                daemon=True,
                name="fuchsia-fill-heartbeat",
            )
            heartbeat_thread.start()
            try:
                rollouts = self.environment.generate(
                    rollouts,
                    self.llm,
                    self._sampling_params,
                    vllm_generate_kwargs=generation_kwargs,
                    tokenizer=self.tokenizer,
                )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)

            payload_samples: list[RolloutSample] = self.environment.payload(
                rollouts,
                tokenizer=self.tokenizer,
            )
            if payload_samples:
                items_with_rewards.extend(sample.to_dict() for sample in payload_samples)
        return items_with_rewards

    def _server_info_payload(self) -> dict:
        payload = {
            "model": self.config.model,
            "mode": "Data Sampler" if self.is_data_sampler else "Standard VLLM",
            "host": self.config.host,
            "port": self.config.port,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype,
            "max_model_len": self.config.max_model_len,
            "buffer_size": getattr(self.config, "buffer_size", None) if self.is_data_sampler else None,
            "dataset_field": getattr(self.config, "dataset_field", None) if self.is_data_sampler else None,
            "sample_transfer_mode": getattr(self, "sample_transfer_mode", None) if self.is_data_sampler else None,
            "sample_transfer_dir": (
                str(self.rollout_queue.queue_dir)
                if (self.is_data_sampler and self.rollout_queue.queue_dir is not None)
                else None
            ),
            "is_sleeping": getattr(self, "_is_sleeping", False),
            "sleep_requested": getattr(self, "_sleep_requested", False),
            "current_buffer_size": self._rollout_queue_size() if self.is_data_sampler else None,
            "is_filling": getattr(self, "_is_filling", False) if self.is_data_sampler else None,
        }
        return payload

    def _buffer_status_payload(self) -> dict:
        now = time.time()
        current_fill_elapsed = None
        if self._is_filling and self._last_fill_started_at is not None:
            current_fill_elapsed = max(0.0, now - self._last_fill_started_at)
        return {
            "current_size": self._rollout_queue_size(),
            "max_size": self.buffer_size,
            "transfer_mode": self.sample_transfer_mode,
            "is_filling": self._is_filling,
            "fill_slot_claimed": self._fill_claim_lock.locked(),
            "is_sleeping": getattr(self, "_is_sleeping", False),
            "sleep_requested": getattr(self, "_sleep_requested", False),
            "generation_in_progress": self._generation_lock.locked(),
            "epoch": self._epoch,
            "last_fill_started_at": self._last_fill_started_at,
            "last_fill_finished_at": self._last_fill_finished_at,
            "last_fill_heartbeat_at": self._last_fill_heartbeat_at,
            "last_fill_duration_seconds": self._last_fill_duration_seconds,
            "last_fill_error": self._last_fill_error,
            "last_fill_generated_groups": self._last_fill_generated_groups,
            "current_fill_elapsed_seconds": current_fill_elapsed,
        }

    def _generate_outputs(
        self,
        prompts: list[str],
        *,
        sampling_params: SamplingParams,
        generation_kwargs: Optional[dict] = None,
    ):
        kwargs = generation_kwargs or {}
        return self.llm.generate(prompts, sampling_params=sampling_params, **kwargs)

    def _generate_with_lock_for_sampler(
        self,
        prompts: list[str],
        *,
        sampling_params: SamplingParams,
    ):
        with self._generation_lock:
            if self._generation_blocked_by_sleep():
                logger.info("Sleep requested - aborting generation")
                return []
            return self._generate_outputs(
                prompts,
                sampling_params=sampling_params,
                generation_kwargs={"lora_path": self.lora_path},
            )

    def _request_sleep(self) -> dict:
        """Puts the LLM engine to sleep, offloading weights to CPU and clearing KV cache."""
        try:
            if self.is_data_sampler:
                generation_in_progress = not self._generation_lock.acquire(blocking=False)
                if generation_in_progress:
                    logger.info("Generation in progress - cannot sleep now")
                    return {
                        "message": "Generation in progress - cannot sleep now",
                        "sleep": False,
                        "sleep_requested": False,
                    }
                try:
                    self._sleep_requested = True
                    logger.info("Sleep requested - proceeding with sleep...")
                    self._is_sleeping = True
                finally:
                    self._generation_lock.release()

            self.llm.sleep(level=1)  # Level 1: offload weights to CPU & clear KV cache
            torch.cuda.synchronize()
            torch.randn(1).cuda()
            torch.cuda.empty_cache()

            if self.is_data_sampler:
                self._sleep_requested = False
            return {"message": "LLM engine has been put to sleep successfully", "sleep": True}
        except Exception as exc:
            logger.error("Failed to put LLM to sleep: %s", exc)
            if self.is_data_sampler:
                self._sleep_requested = False
                self._is_sleeping = False
            return {"error": f"Failed to put LLM to sleep: {exc}", "sleep": False}

    async def _request_wake_up(self) -> dict:
        """Wakes up the LLM engine from sleep mode."""
        try:
            if self.is_data_sampler and not self._is_sleeping:
                return {
                    "message": "LLM engine is already awake",
                    "wake_up": True,
                    "already_awake": True,
                }
            self.llm.wake_up()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.randn(1).cuda()
            await asyncio.sleep(1)
            if self.is_data_sampler:
                # Rotate LoRA request id once per wake cycle (policy step),
                # instead of per generation call, to avoid adapter churn.
                self._lora_request_id += 1
                self._sleep_requested = False
                self._is_sleeping = False
            return {"message": "LLM engine has been woken up successfully", "wake_up": True}
        except Exception as exc:
            logger.error("Failed to wake up LLM: %s", exc)
            return {"error": f"Failed to wake up LLM: {exc}", "wake_up": False}

    def _try_schedule_buffer_fill(
        self,
        *,
        reason: str | None = None,
    ) -> bool:
        """
        Schedule exactly one outstanding buffer fill worker.
        Returns True when a new fill worker was started.
        """
        if not self.is_data_sampler:
            return False
        if self._sleep_block_reason() is not None:
            return False
        if self._rollout_queue_size() >= self.buffer_size:
            return False

        # Claim slot at enqueue time so duplicate background tasks cannot pile up.
        if not self._fill_claim_lock.acquire(blocking=False):
            return False

        # Start a dedicated daemon worker so request lifecycle/cancellation cannot
        # drop the claimed slot and strand the buffer in a "claimed forever" state.
        worker = threading.Thread(
            target=self.buffer_fill,
            kwargs={"claim_acquired": True},
            daemon=True,
            name="fuchsia-buffer-fill",
        )
        try:
            worker.start()
        except Exception:
            self._fill_claim_lock.release()
            raise

        if reason:
            logger.debug("Scheduled buffer fill (%s)", reason)
        return True

    def _create_app(self) -> FastAPI:
        app = FastAPI()

        @app.on_event("startup")
        async def startup():
            if self._prefill_buffer_on_startup:
                threading.Thread(
                    target=self.buffer_fill,
                    daemon=True,
                    name="fuchsia-buffer-prefill",
                ).start()
                logger.info("Started background initial buffer fill")

        @app.get("/")
        async def root():
            return {
                "service": "fuchsia-vllm-server",
                "status": "ok",
                "health": "/health/",
                "server_info": "/server_info/",
            }

        @app.get("/health/")
        async def health():
            """Health check endpoint to verify that the server is running."""
            return {"status": "ok"}

        @app.get("/get_tensor_parallel_size/")
        async def get_tensor_parallel_size():
            """Retrieves the tensor parallel size from the LLM engine."""
            return {
                "tensor_parallel_size": self.llm.llm_engine.parallel_config.tensor_parallel_size
            }

        @app.get("/server_info/")
        async def server_info():
            """Returns basic server configuration information."""
            return self._server_info_payload()

        @app.post("/generate/", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Generates completions for the provided prompts."""
            guided_decoding = None
            if request.guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=request.guided_decoding_regex
                )

            sampling_params = SamplingParams(
                n=request.n,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                max_tokens=request.max_tokens,
                guided_decoding=guided_decoding,
                stop=self.stop,
            )

            if self.is_data_sampler:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    all_outputs = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: self._generate_with_lock_for_sampler(
                            request.prompts,
                            sampling_params=sampling_params,
                        ),
                    )
            else:
                all_outputs = self._generate_outputs(
                    request.prompts,
                    sampling_params=sampling_params,
                    generation_kwargs={"lora_path": self.lora_path},
                )
            
            completion_ids = [
                list(output.token_ids)
                for outputs in all_outputs
                for output in outputs.outputs
            ]
            return {"completion_ids": completion_ids}

        @app.post("/init_communicator/")
        async def init_communicator(
            request: InitCommunicatorRequest, background_tasks: BackgroundTasks
        ):
            """Initializes the communicator for weight synchronization."""
            background_tasks.add_task(
                self.llm.collective_rpc,
                "init_communicator",
                args=(request.host, request.port, self.config.tensor_parallel_size + 1),
            )
            return {"message": "Request received, initializing communicator"}

        @app.post("/update_named_param/")
        async def update_named_param(
            request: UpdateWeightsRequest, background_tasks: BackgroundTasks
        ):
            """Updates model weights with the provided tensor."""
            dtype = torch.__getattribute__(request.dtype.split(".")[-1])
            background_tasks.add_task(
                self.llm.collective_rpc,
                "update_named_param",
                args=(request.name, dtype, request.shape),
            )
            return {"message": "Request received, updating named parameter"}

        @app.post("/reset_prefix_cache/")
        async def reset_prefix_cache():
            """Resets the prefix cache for the model."""
            success = self.llm.llm_engine.reset_prefix_cache()
            return {"message": f"Reset prefix cache status: {success}"}

        @app.post("/close_communicator/")
        async def close_communicator():
            """Closes the weight update group and cleans up resources."""
            self.llm.collective_rpc("close_communicator")
            return {"message": "Request received, closing communicator"}

        @app.post("/sleep/")
        async def sleep():
            return self._request_sleep()

        @app.post("/wake_up/")
        async def wake_up():
            return await self._request_wake_up()

        # Add data sampler specific endpoints if in data sampler mode
        if self.is_data_sampler:
            @app.post("/get_sample/")
            async def get_sample():
                """Returns a sample from the buffer and triggers background buffer fill."""
                if self._rollout_queue_size() == 0:
                    self._try_schedule_buffer_fill(
                        reason="queue-empty",
                    )
                    await asyncio.sleep(1)
                    return {"sample": None}
                items = self._rollout_queue_pop()
                if items is None:
                    return {"sample": None}
                if self._try_schedule_buffer_fill(
                    reason="low-watermark",
                ):
                    logger.info("requesting buffer fill")
                else:
                    reason = self._sleep_block_reason()
                    if reason is not None:
                        logger.info("Skipping buffer fill request - %s", reason)
                return {"sample": items}

            @app.post("/buffer_fill/")
            async def buffer_fill():
                """Fills the buffer with new samples if not already filling."""
                sleep_reason = self._sleep_block_reason()
                if sleep_reason is not None:
                    return {"message": f"Buffer fill skipped - {sleep_reason}"}
                if self._rollout_queue_size() >= self.buffer_size:
                    return {"message": "Buffer already full"}
                if not self._try_schedule_buffer_fill(reason="manual-request"):
                    return {"message": "Buffer fill already in progress"}
                return {"message": "Buffer filling started"}

            @app.get("/buffer_status/")
            async def buffer_status():
                """Returns the current status of the buffer."""
                return self._buffer_status_payload()

            @app.post("/empty_buffer/")
            async def empty_buffer():
                """Empties the buffer and returns the number of items removed."""
                items_removed = self._rollout_queue_clear()
                return {
                    "message": "Buffer emptied successfully",
                    "items_removed": items_removed,
                }

        return app

    def buffer_fill(self, claim_acquired: bool = False):
        """Fills the buffer with new samples."""
        if not self.is_data_sampler:
            return
        with self._fill_claim(claim_acquired) as has_claim:
            if not has_claim:
                return

            sleep_reason = self._sleep_block_reason()
            if sleep_reason is not None:
                logger.info("Skipping buffer fill - %s", sleep_reason)
                return

            with self._generation_lock:
                if self._generation_blocked_by_sleep():
                    logger.info("Sleep requested - aborting buffer fill")
                    return

                fill_started_at = self._start_fill_metrics()
                try:
                    self._fill_rollout_queue()
                except Exception as exc:
                    self._last_fill_error = str(exc)
                    raise
                finally:
                    self._finish_fill_metrics(fill_started_at)

    def _fill_rollout_queue(self) -> None:
        consecutive_empty_batches = 0
        max_empty_batches = max(8, self.buffer_size * 2)
        generation_kwargs = self._current_generation_kwargs()
        max_items_per_generate_call = 2 if self.config.single_gpu else self._generation_batch_size
        max_items_per_generate_call = max(1, int(max_items_per_generate_call))

        while self._rollout_queue_size() < self.buffer_size:
            self._last_fill_heartbeat_at = time.time()
            if self._generation_blocked_by_sleep():
                logger.info("Sleep requested during buffer fill - stopping buffer fill")
                break

            logger.debug("Buffer Size: %s", self._rollout_queue_size())
            items = self._next_dataset_items(self._generation_batch_size)
            if not items:
                raise RuntimeError("Buffer fill produced no source items")

            start_time = time.perf_counter()
            items_with_rewards = self._environment_payload_for_items(
                items,
                generation_kwargs=generation_kwargs,
                max_items_per_generate_call=max_items_per_generate_call,
            )
            logger.debug(
                "buffer fill generation time: %.3fs",
                time.perf_counter() - start_time,
            )

            if not items_with_rewards:
                consecutive_empty_batches += 1
                logger.warning(
                    "Buffer fill generated empty reward payload (%s/%s)",
                    consecutive_empty_batches,
                    max_empty_batches,
                )
                if consecutive_empty_batches >= max_empty_batches:
                    self._last_fill_error = (
                        "Buffer fill repeatedly produced zero payload batches"
                    )
                    break
                continue

            consecutive_empty_batches = 0
            logger.debug("==========")
            self._rollout_queue_push(items_with_rewards)
            self._last_fill_generated_groups += len(items_with_rewards)
            self._last_fill_heartbeat_at = time.time()
            logger.debug("rollout queue size: %s", self._rollout_queue_size())

    def _fill_heartbeat_loop(self, stop_event: threading.Event, interval: float = 2.0) -> None:
        """Background heartbeat for long-running fill generations."""
        while not stop_event.wait(interval):
            self._last_fill_heartbeat_at = time.time()

    def serve(self):
        """Starts the FastAPI server with rich console output."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Create network info table
        table = Table(
            title="Server Network Information",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Interface", style="cyan")
        table.add_column("URL", style="green")

        for ip_info in get_ip_addresses():
            if ip_info["type"] == "IPv4" and ip_info["interface"] != "lo":
                url = f"http://{ip_info['ip']}:{self.config.port}"
                table.add_row(ip_info["interface"], url)

        # Create server status panel
        mode = "Data Sampler" if self.is_data_sampler else "Standard VLLM"
        status_text = f"Mode: {mode}\nModel: {self.config.model}\nPort: {self.config.port}"
        if self.is_data_sampler:
            status_text += f"\nBuffer Size: {self.buffer_size}"
            status_text += f"\nDataset Field: {self.dataset_field}"
            status_text += f"\nSample Transfer: {self.sample_transfer_mode}"
            if self.rollout_queue.queue_dir is not None:
                status_text += f"\nSample Queue Dir: {self.rollout_queue.queue_dir}"

        # Display information
        console.print("\n")
        console.print(Panel(table, title="[bold]Available Network Interfaces[/bold]"))
        console.print(Panel(status_text, title="[bold]Server Configuration[/bold]"))
        console.print(f"\n[bold blue]Server running on port {self.config.port}[/bold blue]\n")

        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
        dist.destroy_process_group()


# Backward-compatible alias for older imports.
DataSamplerServer = VLLMServer


def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to load")
    parser.add_argument("--revision", type=str)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max_model_len", default=1024, type=int)
    parser.add_argument("--enable_prefix_caching", default=False, type=bool)
    parser.add_argument("--config", type=str, default="examples/vllm_server_config.yaml")

    args = parser.parse_args()

    if args.config:
        config = FuchsiaConfig.from_yaml(args.config)
        for key, value in vars(args).items():
            if value is not None and key != "config":
                setattr(config, key, value)
    else:
        config = FuchsiaConfig(**vars(args))

    if not config.model:
        parser.error("Model must be specified either through --model argument or in config file")

    server = VLLMServer(config)
    server.serve()


if __name__ == "__main__":
    run_server()
