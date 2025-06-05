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

import argparse
import ctypes
import logging
import os
import yaml
from dataclasses import dataclass
from typing import Optional, Sequence, Callable

import torch
import torch.distributed as dist
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
import uvicorn
from collections import defaultdict
import time

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


import asyncio
import threading

# Check CUDA availability
try:
    ctypes.CDLL("libcuda.so.1")
    libcuda_available = True
except OSError:
    libcuda_available = False

if libcuda_available:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
else:
    Worker = object

logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


import socket
import psutil
import numpy as np


def get_ip_addresses():
    """Get all network IP addresses with detailed information"""
    ip_info = []

    # Get hostname
    hostname = socket.gethostname()
    ip_info.append({"interface": "Hostname", "ip": hostname, "type": "System"})

    # Get all network interfaces
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4 addresses
                ip_info.append(
                    {
                        "interface": interface,
                        "ip": addr.address,
                        "type": "IPv4",
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast if addr.broadcast else "N/A",
                    }
                )
            elif addr.family == socket.AF_INET6:  # IPv6 addresses
                ip_info.append(
                    {
                        "interface": interface,
                        "ip": addr.address,
                        "type": "IPv6",
                        "netmask": addr.netmask,
                        "broadcast": "N/A",
                    }
                )

    return ip_info


class WeightSyncWorker(Worker):
    """
    A vLLM worker that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    def __init__(self, *args, **kwargs):
        if not libcuda_available:
            raise ImportError("CUDA is required to use the WeightSyncWorker")
        super().__init__(*args, **kwargs)
        self.pynccl_comm = None
        self.client_rank = None

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


@dataclass
class ServerConfig:
    model: str
    revision: Optional[str] = None
    tensor_parallel_size: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.5
    dtype: str = "auto"
    max_model_len: Optional[int] = 512
    enable_prefix_caching: Optional[bool] = None
    quantization: Optional[str] = None
    
    # Data sampler specific configs
    dataset_field: str = "text"
    buffer_size: int = 32
    enable_lora: bool = False
    lora_path: str = "lora_weights"
    vllm_n: int = 1
    vllm_repetition_penalty: float = 1.0
    vllm_temperature: float = 0.9
    vllm_top_p: float = 1.0
    vllm_top_k: int = -1
    vllm_min_p: float = 0.0
    vllm_max_tokens: int = 1024
    vllm_kv_quantization: bool = False
    generation_batch_size: int = 1
    
    single_gpu: bool = False
       

    def __post_init__(self,**kwargs):
        # Allow for dynamic attribute setting from config files
        for key, value in self.__dict__.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


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
        stop: Optional[list[str]] = None,
    ):
        self.config = config
        logger.info(config)
        self.stop = stop
        
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
            worker_cls="fuchsia.vllm_server.WeightSyncWorker", 
            **kwargs
        )

        # Data sampler specific initialization
        self.dataset = dataset
        self.is_data_sampler = dataset is not None
        self._lora_idx = 0
        
        if self.is_data_sampler:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model)
            self.dataset_field = config.dataset_field
            self.reward_functions = reward_functions or []
            self.buffer_size = config.buffer_size
            self.buffer = []
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
                max_tokens=config.vllm_max_tokens,
                stop=self.stop,
            )
            self.enable_lora = config.enable_lora
            self.lora_path = config.lora_path
            self._is_filling = False
            self._is_sleeping = False  # Track sleep state
            self._sleep_requested = False  # Track if sleep has been requested
            self._generation_lock = threading.Lock()  # Lock for generation operations
            
            if pre_fill_buffer:
                self.buffer_fill()

        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI()

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
                    # Check if generation is currently in progress without blocking
                    generation_in_progress = not self._generation_lock.acquire(blocking=False)
                    if generation_in_progress:
                        logger.info("Generation in progress - cannot sleep now")
                        return {
                            "message": "Generation in progress - cannot sleep now", 
                            "sleep": False
                        }
                    
                    try:
                        # Signal that sleep has been requested
                        self._sleep_requested = True
                        logger.info("Sleep requested - proceeding with sleep...")
                        self._is_sleeping = True
                    finally:
                        self._generation_lock.release()
                
                self.llm.sleep(level=1)  # Level 1: offload weights to CPU & clear KV cache
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
                await asyncio.sleep(5)
                if self.is_data_sampler:
                    self._is_sleeping = False
                return {"message": "LLM engine has been woken up successfully"}
            except Exception as e:
                logger.error(f"Failed to wake up LLM: {e}")
                return {"error": f"Failed to wake up LLM: {str(e)}"}

        # Add data sampler specific endpoints if in data sampler mode
        if self.is_data_sampler:
            @app.post("/get_sample/")
            async def get_sample(background_tasks: BackgroundTasks):
                """Returns a sample from the buffer and triggers background buffer fill."""
                if len(self.buffer) == 0:
                    await asyncio.sleep(5)
                    return {"sample": None}
                items = self.buffer.pop(0)
                # Only trigger buffer fill if not sleeping and sleep not requested
                if (len(self.buffer) < self.buffer_size and 
                    not getattr(self, '_is_sleeping', False) and 
                    not getattr(self, '_sleep_requested', False)):
                    background_tasks.add_task(self.buffer_fill)
                    logger.info("requesting buffer fill")
                elif getattr(self, '_is_sleeping', False):
                    logger.info("Skipping buffer fill request - LLM is sleeping")
                elif getattr(self, '_sleep_requested', False):
                    logger.info("Skipping buffer fill request - sleep requested")
                return {"sample": items}

            @app.post("/buffer_fill/")
            async def buffer_fill(background_tasks: BackgroundTasks):
                """Fills the buffer with new samples if not already filling."""
                if self._is_filling:
                    return {"message": "Buffer fill already in progress"}
                if getattr(self, '_is_sleeping', False):
                    return {"message": "Buffer fill skipped - LLM is sleeping"}
                if getattr(self, '_sleep_requested', False):
                    return {"message": "Buffer fill skipped - sleep requested"}
                background_tasks.add_task(self.buffer_fill)
                return {"message": "Buffer filling started"}

            @app.get("/buffer_status/")
            async def buffer_status():
                """Returns the current status of the buffer."""
                return {
                    "current_size": len(self.buffer),
                    "max_size": self.buffer_size,
                    "is_filling": self._is_filling,
                    "is_sleeping": getattr(self, '_is_sleeping', False),
                    "sleep_requested": getattr(self, '_sleep_requested', False),
                    "epoch": self._epoch,
                }

            @app.post("/empty_buffer/")
            async def empty_buffer():
                """Empties the buffer and returns the number of items removed."""
                items_removed = len(self.buffer)
                self.buffer.clear()
                return {
                    "message": "Buffer emptied successfully",
                    "items_removed": items_removed,
                }

        return app

    def buffer_fill(self):
        """Fills the buffer with new samples."""
        if not self.is_data_sampler or self._is_filling:
            return
        
        # Don't fill buffer if LLM is sleeping or sleep is requested
        if (hasattr(self, '_is_sleeping') and self._is_sleeping) or \
           (hasattr(self, '_sleep_requested') and self._sleep_requested):
            logger.info("Skipping buffer fill - LLM is sleeping or sleep requested")
            return

        self._is_filling = True
        try:
            while len(self.buffer) < self.buffer_size:
                # Check if sleep was requested during buffer fill
                if hasattr(self, '_sleep_requested') and self._sleep_requested:
                    logger.info("Sleep requested during buffer fill - stopping buffer fill")
                    break
                    
                items = []
                for _ in range(self._generation_batch_size):
                    try:
                        item = next(self.dataset_iter)
                        items.append(item)
                    except StopIteration:
                        self.dataset_iter = iter(self.dataset)
                        self._epoch += 1

                start_time = time.perf_counter()
                items_with_rewards = self.process_sample(items)
                end_time = time.perf_counter()
                print(f"time taken: {end_time - start_time}")
                print("==========")
                for item in items_with_rewards:
                    print(f"{item['all_rewards']}")
                    print(f"{item['mean']} {item['std']}")
                    print("-"*10)
                logger.debug("==========")
                self.buffer.extend(items_with_rewards)
                logger.debug(f"buffer: {len(self.buffer[0]['completions'])}")
        finally:
            self._is_filling = False

    def process_sample(self, items):
        """Processes samples and calculates rewards."""
        if not self.is_data_sampler:
            return []

        with self._generation_lock:
            # Check if sleep was requested while waiting for the lock
            if hasattr(self, '_sleep_requested') and self._sleep_requested:
                logger.info("Sleep requested - aborting sample processing")
                return []
                
            prompts = [item[self.dataset_field] for item in items]
            if self.config.single_gpu and os.path.exists(self.lora_path):
                all_outputs = self.llm.generate(prompts, sampling_params=self._sampling_params, lora_request=LoRARequest("grpo", self._lora_idx, self.lora_path))
                self._lora_idx += 1
            else:
                all_outputs = self.llm.generate(prompts, sampling_params=self._sampling_params)
            
            completion_ids = [
                list(output.token_ids)
                for outputs in all_outputs
                for output in outputs.outputs
            ]
            stop_reason = [output.stop_reason for outputs in all_outputs for output in outputs.outputs]
            finish_reason = [output.finish_reason for outputs in all_outputs for output in outputs.outputs]
            completions = [self.tokenizer.decode(c) for c in completion_ids]

            all_outputs = []
            for g_idx, item in enumerate(items):
                output = {
                    "item": [item] * self.config.vllm_n,
                    "completions": [],
                    "completion_ids": [],
                    "stop_reason": [],
                    "finish_reason": [],
                    "epoch": self._epoch,
                    "inputs": item[self.dataset_field]
                }
                
                for idx in range(self.config.vllm_n):
                    base_idx = g_idx * self.config.vllm_n + idx
                    output["completions"].append(completions[base_idx])
                    output["completion_ids"].append(completion_ids[base_idx])
                    output["stop_reason"].append(stop_reason[base_idx])
                    output["finish_reason"].append(finish_reason[base_idx])

                output["all_rewards"], output["rewards"], output["mean"], output["std"] = (
                    self.calculate_rewards(
                        output["item"], output["completions"], output["completion_ids"]
                    )
                )
                all_outputs.append(output)

            return all_outputs

    def calculate_rewards(self, items, completions, completion_ids):
        """Calculates rewards for generated completions."""
        if not self.is_data_sampler:
            return {}, [], 0.0, 0.0

        all_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function(
                self.tokenizer, items, completions, completion_ids
            )
            all_rewards[reward_function.__name__] = rewards

        reward_values = np.array([list(rewards) for rewards in all_rewards.values()])
        total_rewards = reward_values.sum(axis=0)
        mean = float(total_rewards.mean())
        std = float(total_rewards.std())

        return all_rewards, total_rewards.tolist(), mean, std

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

        # Display information
        console.print("\n")
        console.print(Panel(table, title="[bold]Available Network Interfaces[/bold]"))
        console.print(Panel(status_text, title="[bold]Server Configuration[/bold]"))
        console.print(f"\n[bold blue]Server running on port {self.config.port}[/bold blue]\n")

        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
        dist.destroy_process_group()


class VLLMClient:
    """Client for interacting with the VLLM server."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        
    def _make_request(self, method: str, endpoint: str, json_data=None):
        """Make HTTP request to the server."""
        import requests
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def health_check(self):
        """Check if the server is healthy."""
        return self._make_request("GET", "/health/")
    
    def generate(self, prompts, **kwargs):
        """Generate completions for the given prompts."""
        request_data = {"prompts": prompts}
        request_data.update(kwargs)
        return self._make_request("POST", "/generate/", request_data)
    
    def get_sample(self):
        """Get a sample from the data sampler buffer."""
        return self._make_request("POST", "/get_sample/")
    
    def get_buffer_status(self):
        """Get the current buffer status."""
        return self._make_request("GET", "/buffer_status/")
    
    def empty_buffer(self):
        """Empty the server buffer."""
        return self._make_request("POST", "/empty_buffer/")
    
    def buffer_fill(self):
        """Trigger buffer fill."""
        return self._make_request("POST", "/buffer_fill/")
    
    def sleep(self):
        """Put the LLM engine to sleep."""
        return self._make_request("POST", "/sleep/")
    
    def wake_up(self):
        """Wake up the LLM engine from sleep."""
        return self._make_request("POST", "/wake_up/")


def load_config_from_yaml(yaml_path: str) -> ServerConfig:
    """Load server configuration from a YAML file."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ServerConfig(**config_dict)


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
        config = load_config_from_yaml(args.config)
        for key, value in vars(args).items():
            if value is not None and key != "config":
                setattr(config, key, value)
    else:
        config = ServerConfig(**vars(args))

    if not config.model:
        parser.error("Model must be specified either through --model argument or in config file")

    server = VLLMServer(config)
    server.serve()


def test_datasampler():
    max_model_len = 1024
    config = ServerConfig(
        model="unsloth/Llama-3.2-3B-Instruct",
        revision="main",
        host="0.0.0.0",
        port=8000,
        dataset_field="Question Text",
        buffer_size=4,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.7,
        dtype="bfloat16",
        vllm_max_tokens=max_model_len,
        vllm_n=8,
        vllm_repetition_penalty=1.0,
        vllm_temperature=1.0,
        vllm_top_p=1.0,
        vllm_top_k=-1,
        vllm_min_p=0.0,
    )
    ds = load_dataset("CK0607/2025-Jee-Mains-Question", split="train")

    def reward_function(tokenizer, items, completions, completion_ids):
        return [len(completion) for completion in completions]

    server = DataSamplerServer(config, dataset=ds, reward_functions=[reward_function])
    server.serve()


if __name__ == "__main__":
    test_datasampler()
