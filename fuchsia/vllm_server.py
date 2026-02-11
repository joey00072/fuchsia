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
import inspect
import logging
import os
import threading
import time
from typing import Optional, Sequence, Callable

# Third party imports
import numpy as np
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
from fuchsia.envs import Rollout, Environment, SingleTurnEnvironment
from fuchsia.reward_utils import clean_completions
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



ServerConfig = FuchsiaConfig


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


class DataSamplerServer:
    def __init__(
        self, 
        config: ServerConfig,
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
            self.reward_functions = reward_functions or []
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
        if getattr(self, "_is_sleeping", False):
            return False
        if getattr(self, "_sleep_requested", False):
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
            return {
                "model": self.config.model,
                "mode": "Data Sampler" if self.is_data_sampler else "Standard VLLM",
                "host": self.config.host,
                "port": self.config.port,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "dtype": self.config.dtype,
                "max_model_len": self.config.max_model_len,
                "buffer_size": getattr(self.config, 'buffer_size', None) if self.is_data_sampler else None,
                "dataset_field": getattr(self.config, 'dataset_field', None) if self.is_data_sampler else None,
                "sample_transfer_mode": getattr(self, "sample_transfer_mode", None) if self.is_data_sampler else None,
                "sample_transfer_dir": str(self.rollout_queue.queue_dir) if (self.is_data_sampler and self.rollout_queue.queue_dir is not None) else None,
                "is_sleeping": getattr(self, '_is_sleeping', False),
                "sleep_requested": getattr(self, '_sleep_requested', False),
                "current_buffer_size": self._rollout_queue_size() if self.is_data_sampler else None,
                "is_filling": getattr(self, '_is_filling', False) if self.is_data_sampler else None,
            }

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
            
            # Use generation lock if in data sampler mode
            if self.is_data_sampler:
                def generate_with_lock():
                    with self._generation_lock:
                        # Check if sleep was requested
                        if hasattr(self, '_sleep_requested') and self._sleep_requested:
                            logger.info("Sleep requested - aborting generation")
                            return []
                        return self.llm.generate(
                            request.prompts, sampling_params=sampling_params,
                            lora_path=self.lora_path,
                        )
                
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    all_outputs = await asyncio.get_event_loop().run_in_executor(
                        executor, generate_with_lock
                    )
            else:
                all_outputs = self.llm.generate(
                    request.prompts, sampling_params=sampling_params,
                    lora_path=self.lora_path,
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
            """Puts the LLM engine to sleep, offloading weights to CPU and clearing KV cache."""
            try:
                if self.is_data_sampler:
                    # Check if generation is currently in progress without blocking.
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
                torch.cuda.empty_cache()  # Clear CUDA cache after sleep
                
                if self.is_data_sampler:
                    self._sleep_requested = False  # Reset the flag
                    
                return {
                    "message": "LLM engine has been put to sleep successfully", 
                    "sleep": True
                }
            except Exception as e:
                logger.error(f"Failed to put LLM to sleep: {e}")
                if self.is_data_sampler:
                    self._sleep_requested = False  # Reset the flag on error
                    self._is_sleeping = False
                return {
                    "error": f"Failed to put LLM to sleep: {str(e)}", 
                    "sleep": False
                }

        @app.post("/wake_up/")
        async def wake_up():
            """Wakes up the LLM engine from sleep mode."""
            try:
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
                return {"message": "LLM engine has been woken up successfully"}
            except Exception as e:
                logger.error(f"Failed to wake up LLM: {e}")
                return {"error": f"Failed to wake up LLM: {str(e)}"}

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
                elif getattr(self, '_is_sleeping', False):
                    logger.info("Skipping buffer fill request - LLM is sleeping")
                elif getattr(self, '_sleep_requested', False):
                    logger.info("Skipping buffer fill request - sleep requested")
                return {"sample": items}

            @app.post("/buffer_fill/")
            async def buffer_fill():
                """Fills the buffer with new samples if not already filling."""
                if getattr(self, '_is_sleeping', False):
                    return {"message": "Buffer fill skipped - LLM is sleeping"}
                if getattr(self, '_sleep_requested', False):
                    return {"message": "Buffer fill skipped - sleep requested"}
                if self._rollout_queue_size() >= self.buffer_size:
                    return {"message": "Buffer already full"}
                if not self._try_schedule_buffer_fill(reason="manual-request"):
                    return {"message": "Buffer fill already in progress"}
                return {"message": "Buffer filling started"}

            @app.get("/buffer_status/")
            async def buffer_status():
                """Returns the current status of the buffer."""
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
                    "is_sleeping": getattr(self, '_is_sleeping', False),
                    "sleep_requested": getattr(self, '_sleep_requested', False),
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

        lock_acquired_here = False
        # Ensure only one buffer fill worker can run or wait-for-lock at a time.
        if not claim_acquired:
            if not self._fill_claim_lock.acquire(blocking=False):
                return
            lock_acquired_here = True

        fill_started_at: float | None = None
        fill_started = False
        # Don't fill buffer if LLM is sleeping or sleep is requested
        if (hasattr(self, '_is_sleeping') and self._is_sleeping) or \
           (hasattr(self, '_sleep_requested') and self._sleep_requested):
            logger.info("Skipping buffer fill - LLM is sleeping or sleep requested")
            if claim_acquired or lock_acquired_here:
                self._fill_claim_lock.release()
            return

        try:
            with self._generation_lock:
                # Check if sleep was requested while waiting for the lock
                if hasattr(self, '_sleep_requested') and self._sleep_requested:
                    logger.info("Sleep requested - aborting buffer fill")
                    return

                self._is_filling = True
                fill_started = True
                fill_started_at = time.time()
                self._last_fill_started_at = fill_started_at
                self._last_fill_heartbeat_at = fill_started_at
                self._last_fill_finished_at = None
                self._last_fill_duration_seconds = None
                self._last_fill_error = None
                self._last_fill_generated_groups = 0
                consecutive_empty_batches = 0
                max_empty_batches = max(8, self.buffer_size * 2)
                try:
                    while self._rollout_queue_size() < self.buffer_size:
                        self._last_fill_heartbeat_at = time.time()
                        # Stop early if sleep was requested during buffer fill.
                        if hasattr(self, '_sleep_requested') and self._sleep_requested:
                            logger.info("Sleep requested during buffer fill - stopping buffer fill")
                            break

                        logger.debug("Buffer Size: %s", self._rollout_queue_size())

                        items = []
                        for _ in range(self._generation_batch_size):
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

                        if not items:
                            raise RuntimeError("Buffer fill produced no source items")

                        start_time = time.perf_counter()

                        generation_kwargs = {}
                        if self.config.single_gpu and os.path.exists(self.lora_path):
                            generation_kwargs["lora_request"] = LoRARequest(
                                "grpo", self._lora_request_id, self.lora_path
                            )
                        items_with_rewards: list[dict] = []
                        # Avoid very large generate() calls (32 prompts) that can intermittently
                        # wedge after many sleep/wake cycles, but keep enough batching for speed.
                        max_items_per_generate_call = 2 if self.config.single_gpu else self._generation_batch_size
                        max_items_per_generate_call = max(1, int(max_items_per_generate_call))
                        for start_idx in range(0, len(items), max_items_per_generate_call):
                            chunk_items = items[start_idx : start_idx + max_items_per_generate_call]
                            rollouts: list[Rollout] = []
                            for item in chunk_items:
                                prompt_text = item[self.dataset_field]
                                prompt_ids = self.tokenizer(
                                    prompt_text, add_special_tokens=False
                                )["input_ids"]
                                rollouts.append(
                                    Rollout(prompt=prompt_text, prompt_ids=prompt_ids, item=item)
                                )

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

                            payload = self.environment.payload(rollouts, tokenizer=self.tokenizer)
                            if payload:
                                items_with_rewards.extend(payload)
                        end_time = time.perf_counter()
                        logger.debug("buffer fill generation time: %.3fs", end_time - start_time)
                        if len(items_with_rewards) == 0:
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
                except Exception as exc:
                    self._last_fill_error = str(exc)
                    raise
                finally:
                    self._is_filling = False
                    fill_finished_at = time.time()
                    self._last_fill_finished_at = fill_finished_at
                    if fill_started_at is not None:
                        self._last_fill_duration_seconds = max(
                            0.0, fill_finished_at - fill_started_at
                        )
        finally:
            if claim_acquired or lock_acquired_here:
                self._fill_claim_lock.release()

    def process_sample(self, items):
        """Processes samples and calculates rewards."""
        if not self.is_data_sampler:
            return []

        # Check if sleep was requested
        if hasattr(self, '_sleep_requested') and self._sleep_requested:
            logger.info("Sleep requested - aborting sample processing")
            return []
            
        prompts = [item[self.dataset_field] for item in items]
        
        
        generation_kwargs = {}
        if self.config.single_gpu and os.path.exists(self.lora_path):
            generation_kwargs["lora_request"] = LoRARequest("grpo", self._lora_request_id, self.lora_path)
        
        print(f">>>>> Generation_kwargs: {generation_kwargs} <<<<<")
        all_outputs = self.llm.generate(prompts, sampling_params=self._sampling_params, **generation_kwargs)
        
        completion_ids = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        completion_logprobs = [
            self.environment._extract_completion_logprobs(output)
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        stop_reason = [output.stop_reason for outputs in all_outputs for output in outputs.outputs]
        finish_reason = [output.finish_reason for outputs in all_outputs for output in outputs.outputs]
        completions = [self.tokenizer.decode(c) for c in completion_ids]

        all_outputs = []
        for g_idx, item in enumerate(items):
            prompt_text = item[self.dataset_field]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            output = {
                "item": [item] * self.config.vllm_n,
                "prompt_ids": prompt_ids,
                "completions": [],
                "completion_ids": [],
                "completion_logprobs": [],
                "stop_reason": [],
                "finish_reason": [],
                "epoch": self._epoch,
                "inputs": prompt_text
            }
            
            for idx in range(self.config.vllm_n):
                base_idx = g_idx * self.config.vllm_n + idx
                output["completions"].append(completions[base_idx])
                output["completion_ids"].append(completion_ids[base_idx])
                output["completion_logprobs"].append(completion_logprobs[base_idx])
                output["stop_reason"].append(stop_reason[base_idx])
                output["finish_reason"].append(finish_reason[base_idx])

            output["all_rewards"], output["rewards"], output["mean"], output["std"] = (
                self.calculate_rewards(
                    output["item"], output["completions"], output["completion_ids"]
                )
            )
            all_outputs.append(output)

        return all_outputs

    def _fill_heartbeat_loop(self, stop_event: threading.Event, interval: float = 2.0) -> None:
        """Background heartbeat for long-running fill generations."""
        while not stop_event.wait(interval):
            self._last_fill_heartbeat_at = time.time()

    def calculate_rewards(self, items, completions, completion_ids):
        if not self.is_data_sampler:
            return {}, [], 0.0, 0.0

        all_rewards = {}
        if not self.reward_functions:
            return {}, [], 0.0, 0.0

        reward_meta = {}
        needs_cleaned = False
        for reward_function in self.reward_functions:
            sig = inspect.signature(reward_function)
            params = sig.parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            reward_meta[reward_function] = (params, accepts_kwargs)
            if accepts_kwargs or "cleaned_completions" in params:
                needs_cleaned = True

        cleaned_completions = None
        if needs_cleaned:
            cleaned_completions = clean_completions(
                completions,
                tokenizer=self.tokenizer,
                token_ids_list=completion_ids,
            )

        for reward_function in self.reward_functions:
            params, accepts_kwargs = reward_meta[reward_function]
            base_kwargs = {
                "tokenizer": self.tokenizer,
                "items": items,
                "completions": completions,
                "completion_ids": completion_ids,
            }
            extra_kwargs = {}
            if cleaned_completions is not None:
                extra_kwargs["cleaned_completions"] = cleaned_completions

            if accepts_kwargs:
                kwargs = {**base_kwargs, **extra_kwargs}
                rewards = reward_function(**kwargs)
            else:
                combined = {**base_kwargs, **extra_kwargs}
                kwargs = {k: v for k, v in combined.items() if k in params}
                if kwargs:
                    rewards = reward_function(**kwargs)
                else:
                    rewards = reward_function(self.tokenizer, items, completions, completion_ids)

            all_rewards[reward_function.__name__] = rewards

        # Convert all reward lists to tensors and stack them
        if all_rewards:
            reward_tensors = []
            for rewards in all_rewards.values():
                reward_tensor = torch.tensor(rewards, dtype=torch.float32)
                reward_tensors.append(reward_tensor)
            
            # Stack tensors if we have multiple reward functions, otherwise use the single tensor
            if len(reward_tensors) > 1:
                reward_values = torch.stack(reward_tensors, dim=0)
                total_rewards = reward_values.sum(dim=0)
            else:
                total_rewards = reward_tensors[0]
            
            mean = total_rewards.mean().item()
            std = total_rewards.std().item()
            
            return all_rewards, total_rewards.tolist(), mean, std
        else:
            return {}, [], 0.0, 0.0

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


# Backward-compatible alias used by CLI/server integrations.
VLLMServer = DataSamplerServer


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
        config = ServerConfig.from_yaml(args.config)
        for key, value in vars(args).items():
            if value is not None and key != "config":
                setattr(config, key, value)
    else:
        config = ServerConfig(**vars(args))

    if not config.model:
        parser.error("Model must be specified either through --model argument or in config file")

    server = DataSamplerServer(config)
    server.serve()


if __name__ == "__main__":
    run_server()
