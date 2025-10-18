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
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable

# Third party imports
import numpy as np
import torch
import torch.distributed as dist
import uvicorn
import yaml
from datasets import Dataset, load_dataset
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
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
from fuchsia.envs import Rollout, Environment, SingleTurnEnvironment, MultiTurnEnvironment
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


@dataclass
class ServerConfig:
    """
    Comprehensive configuration for VLLM server with data sampling capabilities.
    
    This configuration class manages settings for:
    - VLLM model server configuration and hardware settings
    - Data sampling and buffer management
    - Generation parameters and optimization settings
    - Dataset loading and processing options
    - LoRA integration and model adaptation
    - Performance tuning and resource management
    
    The class provides extensive validation, documentation, and metadata
    for each configuration option to ensure robust server operation.
    """
    
    # ============================================================================
    # MODEL AND SERVER CONFIGURATION
    # ============================================================================
    
    model: str = field(
        metadata={
            "description": "Model name or path for VLLM server",
            "examples": ["microsoft/DialoGPT-medium", "meta-llama/Llama-2-7b-hf"],
            "required": True,
            "category": "model"
        }
    )
    
    revision: Optional[str] = field(
        default=None,
        metadata={
            "description": "Specific model revision/commit to use",
            "examples": ["main", "v1.0", "abc123def456"],
            "category": "model"
        }
    )
    
    tensor_parallel_size: int = field(
        default=1,
        metadata={
            "description": "Number of GPUs to use for tensor parallelism",
            "range": (1, 8),
            "category": "hardware",
            "impact": "Higher values enable larger models but require more GPUs"
        }
    )
    
    host: str = field(
        default="0.0.0.0",
        metadata={
            "description": "Host address to bind the server to",
            "examples": ["0.0.0.0", "localhost", "127.0.0.1"],
            "category": "network",
            "security_note": "0.0.0.0 allows external connections"
        }
    )
    
    port: int = field(
        default=8000,
        metadata={
            "description": "Port number for the server",
            "range": (1024, 65535),
            "category": "network"
        }
    )
    
    gpu_memory_utilization: float = field(
        default=0.5,
        metadata={
            "description": "Fraction of GPU memory to use for model",
            "range": (0.1, 0.95),
            "category": "hardware",
            "recommendations": {
                "conservative": 0.5,
                "aggressive": 0.8,
                "maximum": 0.9
            }
        }
    )
    
    dtype: str = field(
        default="auto",
        metadata={
            "description": "Data type for model weights",
            "choices": ["auto", "bfloat16", "float16", "float32"],
            "category": "model",
            "usage": "auto = let VLLM decide based on model and hardware"
        }
    )
    
    max_model_len: Optional[int] = field(
        default=512,
        metadata={
            "description": "Maximum sequence length for the model",
            "range": (1, 32768),
            "category": "model",
            "unit": "tokens",
            "impact": "Higher values use more memory but allow longer sequences"
        }
    )
    
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "description": "Enable prefix caching for improved performance",
            "category": "performance",
            "usage": "None = auto-detect, True/False = force enable/disable"
        }
    )
    
    quantization: Optional[str] = field(
        default=None,
        metadata={
            "description": "Quantization method to reduce memory usage",
            "choices": [None, "awq", "gptq", "squeezellm", "fp8"],
            "category": "optimization",
            "trade_off": "Reduces memory but may affect quality"
        }
    )
    
    # ============================================================================
    # DATA SAMPLING CONFIGURATION
    # ============================================================================
    
    dataset_field: str = field(
        default="text",
        metadata={
            "description": "Field name in dataset containing input text",
            "examples": ["text", "prompt", "input", "question"],
            "category": "dataset"
        }
    )
    
    buffer_size: int = field(
        default=32,
        metadata={
            "description": "Size of the data buffer for sampling",
            "range": (1, 1000),
            "category": "dataset",
            "impact": "Larger buffers provide more variety but use more memory"
        }
    )
    
    generation_batch_size: int = field(
        default=1,
        metadata={
            "description": "Batch size for generation during data sampling",
            "range": (1, 64),
            "category": "performance"
        }
    )
    
    dataset_name: str = field(
        default="",
        metadata={
            "description": "Name of the dataset to load",
            "examples": ["squad", "wikitext", "custom_dataset"],
            "category": "dataset"
        }
    )
    
    dataset_split: str = field(
        default="train",
        metadata={
            "description": "Dataset split to use",
            "choices": ["train", "validation", "test"],
            "category": "dataset"
        }
    )
    
    dataset_max_samples: int = field(
        default=-1,
        metadata={
            "description": "Maximum number of samples to load from dataset",
            "range": (-1, 1000000),
            "category": "dataset",
            "usage": "-1 = load all samples"
        }
    )
    
    # ============================================================================
    # LORA CONFIGURATION
    # ============================================================================
    
    enable_lora: bool = field(
        default=False,
        metadata={
            "description": "Enable LoRA adapter support",
            "category": "lora"
        }
    )
    
    lora_path: str = field(
        default="lora_weights",
        metadata={
            "description": "Path to LoRA weights directory",
            "category": "lora"
        }
    )
    
    # ============================================================================
    # VLLM GENERATION PARAMETERS
    # ============================================================================
    
    vllm_n: int = field(
        default=1,
        metadata={
            "description": "Number of completions to generate per prompt",
            "range": (1, 20),
            "category": "generation"
        }
    )
    
    vllm_repetition_penalty: float = field(
        default=1.0,
        metadata={
            "description": "Repetition penalty for VLLM generation",
            "range": (0.5, 2.0),
            "category": "generation"
        }
    )
    
    vllm_temperature: float = field(
        default=0.9,
        metadata={
            "description": "Sampling temperature for VLLM generation",
            "range": (0.01, 2.0),
            "category": "generation"
        }
    )
    
    vllm_top_p: float = field(
        default=1.0,
        metadata={
            "description": "Top-p (nucleus) sampling parameter",
            "range": (0.0, 1.0),
            "category": "generation"
        }
    )
    
    vllm_top_k: int = field(
        default=-1,
        metadata={
            "description": "Top-k sampling parameter",
            "range": (-1, 1000),
            "category": "generation",
            "usage": "-1 = disabled"
        }
    )
    
    vllm_min_p: float = field(
        default=0.0,
        metadata={
            "description": "Minimum probability threshold",
            "range": (0.0, 1.0),
            "category": "generation"
        }
    )
    
    vllm_max_tokens: int = field(
        default=1024,
        metadata={
            "description": "Maximum tokens to generate",
            "range": (1, 4096),
            "category": "generation"
        }
    )
    
    vllm_kv_quantization: bool = field(
        default=False,
        metadata={
            "description": "Enable KV cache quantization to save memory",
            "category": "optimization",
            "trade_off": "Saves memory but may reduce quality"
        }
    )
    
    # ============================================================================
    # SYSTEM CONFIGURATION
    # ============================================================================
    
    single_gpu: bool = field(
        default=False,
        metadata={
            "description": "Force single GPU usage",
            "category": "hardware"
        }
    )
       

    def __post_init__(self, **kwargs):
        """
        Post-initialization validation and setup for ServerConfig.
        
        Performs comprehensive validation of server configuration parameters,
        validates network settings, and ensures compatibility between options.
        """
        # Allow for dynamic attribute setting from config files
        for key, value in self.__dict__.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
                
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate server configuration parameters."""
        errors = []
        
        # Model validation
        if not self.model.strip():
            errors.append("model cannot be empty")
            
        # Network validation
        if not 1024 <= self.port <= 65535:
            errors.append(f"port must be in range [1024, 65535], got {self.port}")
            
        # Hardware validation
        if self.tensor_parallel_size <= 0:
            errors.append(f"tensor_parallel_size must be positive, got {self.tensor_parallel_size}")
            
        if not 0.1 <= self.gpu_memory_utilization <= 0.95:
            errors.append(f"gpu_memory_utilization must be in [0.1, 0.95], got {self.gpu_memory_utilization}")
            
        # Model configuration validation
        if self.max_model_len is not None and self.max_model_len <= 0:
            errors.append(f"max_model_len must be positive, got {self.max_model_len}")
            
        # Data sampling validation
        if self.buffer_size <= 0:
            errors.append(f"buffer_size must be positive, got {self.buffer_size}")
            
        if self.generation_batch_size <= 0:
            errors.append(f"generation_batch_size must be positive, got {self.generation_batch_size}")
            
        if self.dataset_max_samples < -1:
            errors.append(f"dataset_max_samples must be >= -1, got {self.dataset_max_samples}")
            
        # Generation parameter validation
        if self.vllm_n <= 0:
            errors.append(f"vllm_n must be positive, got {self.vllm_n}")
            
        if not 0.01 <= self.vllm_temperature <= 2.0:
            errors.append(f"vllm_temperature must be in [0.01, 2.0], got {self.vllm_temperature}")
            
        if not 0.0 <= self.vllm_top_p <= 1.0:
            errors.append(f"vllm_top_p must be in [0.0, 1.0], got {self.vllm_top_p}")
            
        if not 0.0 <= self.vllm_min_p <= 1.0:
            errors.append(f"vllm_min_p must be in [0.0, 1.0], got {self.vllm_min_p}")
            
        if self.vllm_max_tokens <= 0:
            errors.append(f"vllm_max_tokens must be positive, got {self.vllm_max_tokens}")
            
        # Dtype validation
        valid_dtypes = ["auto", "bfloat16", "float16", "float32"]
        if self.dtype not in valid_dtypes:
            errors.append(f"dtype must be one of {valid_dtypes}, got '{self.dtype}'")
            
        # Dataset split validation
        valid_splits = ["train", "validation", "test"]
        if self.dataset_split not in valid_splits:
            errors.append(f"dataset_split must be one of {valid_splits}, got '{self.dataset_split}'")
            
        # Quantization validation
        valid_quantization = [None, "awq", "gptq", "squeezellm", "fp8"]
        if self.quantization not in valid_quantization:
            errors.append(f"quantization must be one of {valid_quantization}, got '{self.quantization}'")
            
        if errors:
            raise ValueError("ServerConfig validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
            
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of key server configuration parameters."""
        return {
            "server": {
                "model": self.model,
                "host": self.host,
                "port": self.port,
                "tensor_parallel_size": self.tensor_parallel_size
            },
            "hardware": {
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "dtype": self.dtype,
                "single_gpu": self.single_gpu
            },
            "generation": {
                "max_tokens": self.vllm_max_tokens,
                "temperature": self.vllm_temperature,
                "top_p": self.vllm_top_p,
                "n_completions": self.vllm_n
            },
            "dataset": {
                "name": self.dataset_name,
                "field": self.dataset_field,
                "buffer_size": self.buffer_size,
                "max_samples": self.dataset_max_samples
            }
        }
        
    def validate_compatibility(self):
        """Check for compatibility issues between configuration options."""
        warnings = []
        
        # Memory usage warnings
        if self.gpu_memory_utilization > 0.8 and self.tensor_parallel_size == 1:
            warnings.append("High GPU memory utilization (>0.8) with single GPU may cause OOM errors")
            
        # Generation parameter warnings
        if self.vllm_temperature < 0.1:
            warnings.append("Very low temperature (<0.1) may produce repetitive text")
            
        if self.vllm_top_p < 0.1:
            warnings.append("Very low top_p (<0.1) may limit text diversity")
            
        # Buffer size warnings
        if self.buffer_size > 100:
            warnings.append("Large buffer size (>100) may use significant memory")
            
        return warnings

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ServerConfig":
        import yaml
        
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
    
        # Extract server configuration
        server_config = config.get("server", {})
        
        # Extract model configuration
        model_config = config.get("model", {})
        
        # Extract dataset configuration
        dataset_config = config.get("dataset", {})
        
        # Extract GRPO configuration for shared values
        grpo_config = config.get("grpo", {})
        
        # Extract nested vllm configuration
        vllm_config = server_config.get("vllm", {})
        
        # Create ServerConfig with all values loaded from YAML
        server_config_obj = cls(
            # Model configuration
            model=model_config.get("name", ""),
            revision=model_config.get("revision"),
            dtype=model_config.get("dtype", "auto"),
            max_model_len=model_config.get("max_model_len", 512),
            
            # Server configuration
            host=server_config.get("host", "0.0.0.0"),
            port=server_config.get("port", 8000),
            gpu_memory_utilization=server_config.get("gpu_memory_utilization", 0.5),
            tensor_parallel_size=server_config.get("tensor_parallel_size", 1),
            enable_prefix_caching=server_config.get("enable_prefix_caching", None),
            quantization=server_config.get("quantization"),
            buffer_size=server_config.get("buffer_size", 32),
            generation_batch_size=server_config.get("generation_batch_size", 1),
            
            # Dataset configuration
            dataset_field=dataset_config.get("field", "text"),
            dataset_name=dataset_config.get("name", ""),
            dataset_split=dataset_config.get("split", "train"),
            dataset_max_samples=dataset_config.get("max_samples", -1),
            
            # GRPO/LoRA shared configuration
            lora_path=grpo_config.get("lora_path", "lora_weights"),
            single_gpu=grpo_config.get("single_gpu", False),
            
            # VLLM configuration
            vllm_n=vllm_config.get("n", 1),
            vllm_repetition_penalty=vllm_config.get("repetition_penalty", 1.0),
            vllm_temperature=vllm_config.get("temperature", 0.9),
            vllm_top_p=vllm_config.get("top_p", 1.0),
            vllm_top_k=vllm_config.get("top_k", -1),
            vllm_min_p=vllm_config.get("min_p", 0.0),
            vllm_max_tokens=vllm_config.get("max_tokens", 1024),
            vllm_kv_quantization=vllm_config.get("kv_quantization", False),
        )

        return server_config_obj
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        result = {}
        for field_name, field_obj in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            result[field_name] = value
        return result
    
    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific configuration field."""
        if field_name not in self.__dataclass_fields__:
            raise ValueError(f"Field '{field_name}' not found in configuration")
            
        field_obj = self.__dataclass_fields__[field_name]
        info = {
            "name": field_name,
            "type": str(field_obj.type),
            "default": field_obj.default if field_obj.default != field_obj.default_factory else field_obj.default_factory(),
            "current_value": getattr(self, field_name)
        }
        
        # Add metadata if available
        if hasattr(field_obj, 'metadata') and field_obj.metadata:
            info.update(field_obj.metadata)
            
        return info
    
    def list_fields_by_category(self, category: str = None) -> List[str]:
        """List all fields, optionally filtered by category."""
        fields = []
        for field_name, field_obj in self.__dataclass_fields__.items():
            if hasattr(field_obj, 'metadata') and field_obj.metadata:
                field_category = field_obj.metadata.get('category', 'uncategorized')
                if category is None or field_category == category:
                    fields.append(field_name)
            elif category is None:
                fields.append(field_name)
        return fields
    
    def get_categories(self) -> List[str]:
        """Get all available configuration categories."""
        categories = set()
        for field_obj in self.__dataclass_fields__.values():
            if hasattr(field_obj, 'metadata') and field_obj.metadata:
                category = field_obj.metadata.get('category', 'uncategorized')
                categories.add(category)
        return sorted(list(categories))
    
    def print_config_summary(self):
        """Print a formatted summary of the configuration."""
        print("\\n" + "="*60)
        print("SERVER CONFIGURATION SUMMARY")
        print("="*60)
        
        categories = self.get_categories()
        for category in categories:
            print(f"\\n[{category.upper()}]")
            print("-" * 40)
            
            fields = self.list_fields_by_category(category)
            for field_name in fields:
                value = getattr(self, field_name)
                field_info = self.get_field_info(field_name)
                description = field_info.get('description', 'No description available')
                
                # Format value display
                if isinstance(value, float):
                    value_str = f"{value:.6g}"
                else:
                    value_str = str(value)
                    
                print(f"  {field_name}: {value_str}")
                print(f"    {description}")
                
        print("\\n" + "="*60)


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
        self._lora_idx = 1
        
        self.environment = environment or SingleTurnEnvironment(reward_functions=reward_functions)
        
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

        @app.get("/", response_class=HTMLResponse)
        async def root():
            """Serves the main HTML page with UI."""
            html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM Server UI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .section { background: white; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
        button { background: #667eea; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #5a6fd8; }
        button.danger { background: #e74c3c; }
        button.danger:hover { background: #c0392b; }
        button.success { background: #27ae60; }
        button.success:hover { background: #229954; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .response { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #667eea; margin: 10px 0; white-space: pre-wrap; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .full-width { grid-column: 1 / -1; }
        .status-indicators { display: flex; gap: 20px; margin-top: 10px; }
        .status-indicator { 
            background: rgba(255,255,255,0.2); 
            padding: 8px 12px; 
            border-radius: 5px; 
            font-weight: bold;
            font-size: 14px;
        }
        .status-indicator.awake { background: rgba(39, 174, 96, 0.3); }
        .status-indicator.sleeping { background: rgba(231, 76, 60, 0.3); }
        .status-indicator.requested { background: rgba(243, 156, 18, 0.3); }
        @media (max-width: 768px) { 
            .grid { grid-template-columns: 1fr; } 
            .status-indicators { flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 vLLM Server Control Panel</h1>
            <p>Model: <span id="model-name">Loading...</span> | Mode: <span id="server-mode">Loading...</span></p>
            <div class="status-indicators">
                <span id="sleep-status" class="status-indicator">💤 Sleep Status: Loading...</span>
                <span id="buffer-status" class="status-indicator">📊 Buffer: Loading...</span>
            </div>
        </div>

        <div class="grid">
            <!-- Server Status -->
            <div class="section">
                <h2>📊 Server Status</h2>
                <div id="server-status" class="status info">Loading server status...</div>
                <button onclick="getHealth()">🔄 Refresh Status</button>
                <button onclick="getServerInfo()">📊 Refresh Server Info</button>
                <button onclick="getTensorParallelSize()">📏 Get Tensor Parallel Size</button>
            </div>

            <!-- Generation -->
            <div class="section">
                <h2>🤖 Text Generation</h2>
                <div class="form-group">
                    <label for="prompts">Prompts (one per line):</label>
                    <textarea id="prompts" rows="4" placeholder="Enter your prompts here...">Hello, how are you?</textarea>
                </div>
                <div class="form-group">
                    <label for="n">Number of completions (n):</label>
                    <input type="number" id="n" value="1" min="1" max="10">
                </div>
                <div class="form-group">
                    <label for="temperature">Temperature:</label>
                    <input type="number" id="temperature" value="1.0" min="0.0" max="2.0" step="0.1">
                </div>
                <div class="form-group">
                    <label for="max_tokens">Max Tokens:</label>
                    <input type="number" id="max_tokens" value="16" min="1" max="2048">
                </div>
                <button onclick="generate()">🚀 Generate</button>
                <div id="generation-response" class="response" style="display: none;"></div>
            </div>

            <!-- Sleep/Wake Control -->
            <div class="section">
                <h2>💤 Sleep Control</h2>
                <button onclick="sleep()" class="danger">😴 Put to Sleep</button>
                <button onclick="wakeUp()" class="success">🌅 Wake Up</button>
                <div id="sleep-response" class="response" style="display: none;"></div>
            </div>

            <!-- Weight Management -->
            <div class="section">
                <h2>⚙️ Weight Management</h2>
                <div class="form-group">
                    <label for="comm-host">Host:</label>
                    <input type="text" id="comm-host" value="localhost">
                </div>
                <div class="form-group">
                    <label for="comm-port">Port:</label>
                    <input type="number" id="comm-port" value="12345">
                </div>
                <div class="form-group">
                    <label for="world-size">World Size:</label>
                    <input type="number" id="world-size" value="2">
                </div>
                <button onclick="initCommunicator()">🔗 Init Communicator</button>
                <button onclick="closeCommunicator()" class="danger">❌ Close Communicator</button>
                <button onclick="resetPrefixCache()">🔄 Reset Prefix Cache</button>
                <div id="weight-response" class="response" style="display: none;"></div>
            </div>
        </div>

        <!-- Data Sampler Controls (if applicable) -->
        <div id="data-sampler-section" class="section full-width" style="display: none;">
            <h2>📊 Data Sampler Controls</h2>
            <div class="grid">
                <div>
                    <button onclick="getSample()">📥 Get Sample</button>
                    <button onclick="bufferFill()">🔄 Fill Buffer</button>
                    <button onclick="getBufferStatus()">📊 Buffer Status</button>
                </div>
                <div>
                    <button onclick="emptyBuffer()" class="danger">🗑️ Empty Buffer</button>
                </div>
            </div>
            <div id="data-sampler-response" class="response" style="display: none;"></div>
        </div>

        <!-- Response Display -->
        <div class="section full-width">
            <h2>📤 Last Response</h2>
            <div id="last-response" class="response">No requests made yet.</div>
        </div>
    </div>

    <script>
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            getHealth();
            getServerInfo();
            getBufferStatus();
            
            // Auto-refresh status every 2 seconds
            setInterval(function() {
                getServerInfo();
            }, 2000);
        });

        // Utility functions
        function showResponse(elementId, data, isError = false) {
            const element = document.getElementById(elementId);
            element.style.display = 'block';
            element.className = 'response ' + (isError ? 'error' : 'success');
            element.textContent = typeof data === 'object' ? JSON.stringify(data, null, 2) : data;
            
            // Also update last response
            document.getElementById('last-response').textContent = typeof data === 'object' ? JSON.stringify(data, null, 2) : data;
        }

        function hideResponse(elementId) {
            document.getElementById(elementId).style.display = 'none';
        }

        // Server Status
        async function getHealth() {
            try {
                const response = await fetch('/health/');
                const data = await response.json();
                document.getElementById('server-status').textContent = 'Server is running - Status: ' + data.status;
                document.getElementById('server-status').className = 'status success';
            } catch (error) {
                document.getElementById('server-status').textContent = 'Error connecting to server: ' + error.message;
                document.getElementById('server-status').className = 'status error';
            }
        }

        async function getServerInfo() {
            try {
                const response = await fetch('/server_info/');
                const data = await response.json();
                document.getElementById('model-name').textContent = data.model;
                document.getElementById('server-mode').textContent = data.mode;
                
                // Update sleep status
                const sleepStatus = document.getElementById('sleep-status');
                if (data.is_sleeping) {
                    sleepStatus.textContent = '💤 Sleep Status: SLEEPING';
                    sleepStatus.className = 'status-indicator sleeping';
                } else if (data.sleep_requested) {
                    sleepStatus.textContent = '💤 Sleep Status: REQUESTED';
                    sleepStatus.className = 'status-indicator requested';
                } else {
                    sleepStatus.textContent = '💤 Sleep Status: AWAKE';
                    sleepStatus.className = 'status-indicator awake';
                }
                
                // Update buffer status
                const bufferStatus = document.getElementById('buffer-status');
                if (data.mode === 'Data Sampler') {
                    const currentSize = data.current_buffer_size || 0;
                    const maxSize = data.buffer_size || 0;
                    const filling = data.is_filling ? ' (Filling...)' : '';
                    bufferStatus.textContent = `📊 Buffer: ${currentSize}/${maxSize}${filling}`;
                    bufferStatus.className = 'status-indicator';
                } else {
                    bufferStatus.textContent = '📊 Buffer: N/A (Standard Mode)';
                    bufferStatus.className = 'status-indicator';
                }
            } catch (error) {
                document.getElementById('model-name').textContent = 'Unknown';
                document.getElementById('server-mode').textContent = 'Unknown';
                document.getElementById('sleep-status').textContent = '💤 Sleep Status: Error';
                document.getElementById('buffer-status').textContent = '📊 Buffer: Error';
            }
        }

        async function getTensorParallelSize() {
            try {
                const response = await fetch('/get_tensor_parallel_size/');
                const data = await response.json();
                showResponse('generation-response', data);
            } catch (error) {
                showResponse('generation-response', 'Error: ' + error.message, true);
            }
        }

        // Generation
        async function generate() {
            const prompts = document.getElementById('prompts').value.split('\\n').filter(p => p.trim());
            const n = parseInt(document.getElementById('n').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const max_tokens = parseInt(document.getElementById('max_tokens').value);

            const requestData = {
                prompts: prompts,
                n: n,
                temperature: temperature,
                max_tokens: max_tokens,
                repetition_penalty: 1.0,
                top_p: 1.0,
                top_k: -1,
                min_p: 0.0
            };

            try {
                const response = await fetch('/generate/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });
                const data = await response.json();
                showResponse('generation-response', data);
            } catch (error) {
                showResponse('generation-response', 'Error: ' + error.message, true);
            }
        }

        // Sleep Control
        async function sleep() {
            try {
                const response = await fetch('/sleep/', { method: 'POST' });
                const data = await response.json();
                showResponse('sleep-response', data);
            } catch (error) {
                showResponse('sleep-response', 'Error: ' + error.message, true);
            }
        }

        async function wakeUp() {
            try {
                const response = await fetch('/wake_up/', { method: 'POST' });
                const data = await response.json();
                showResponse('sleep-response', data);
            } catch (error) {
                showResponse('sleep-response', 'Error: ' + error.message, true);
            }
        }

        // Weight Management
        async function initCommunicator() {
            const host = document.getElementById('comm-host').value;
            const port = parseInt(document.getElementById('comm-port').value);
            const worldSize = parseInt(document.getElementById('world-size').value);

            const requestData = { host, port, world_size: worldSize };

            try {
                const response = await fetch('/init_communicator/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });
                const data = await response.json();
                showResponse('weight-response', data);
            } catch (error) {
                showResponse('weight-response', 'Error: ' + error.message, true);
            }
        }

        async function closeCommunicator() {
            try {
                const response = await fetch('/close_communicator/', { method: 'POST' });
                const data = await response.json();
                showResponse('weight-response', data);
            } catch (error) {
                showResponse('weight-response', 'Error: ' + error.message, true);
            }
        }

        async function resetPrefixCache() {
            try {
                const response = await fetch('/reset_prefix_cache/', { method: 'POST' });
                const data = await response.json();
                showResponse('weight-response', data);
            } catch (error) {
                showResponse('weight-response', 'Error: ' + error.message, true);
            }
        }

        // Data Sampler Functions
        async function getSample() {
            try {
                const response = await fetch('/get_sample/', { method: 'POST' });
                const data = await response.json();
                showResponse('data-sampler-response', data);
            } catch (error) {
                showResponse('data-sampler-response', 'Error: ' + error.message, true);
            }
        }

        async function bufferFill() {
            try {
                const response = await fetch('/buffer_fill/', { method: 'POST' });
                const data = await response.json();
                showResponse('data-sampler-response', data);
            } catch (error) {
                showResponse('data-sampler-response', 'Error: ' + error.message, true);
            }
        }

        async function getBufferStatus() {
            try {
                const response = await fetch('/buffer_status/');
                const data = await response.json();
                
                // Show data sampler section if buffer status is available
                if (response.ok) {
                    document.getElementById('data-sampler-section').style.display = 'block';
                    document.getElementById('server-mode').textContent = 'Data Sampler';
                    
                    const statusText = `Buffer: ${data.current_size}/${data.max_size} | Filling: ${data.is_filling} | Sleeping: ${data.is_sleeping} | Epoch: ${data.epoch}`;
                    showResponse('data-sampler-response', { ...data, status: statusText });
                } else {
                    document.getElementById('server-mode').textContent = 'Standard VLLM';
                }
            } catch (error) {
                document.getElementById('server-mode').textContent = 'Standard VLLM';
            }
        }

        async function emptyBuffer() {
            try {
                const response = await fetch('/empty_buffer/', { method: 'POST' });
                const data = await response.json();
                showResponse('data-sampler-response', data);
            } catch (error) {
                showResponse('data-sampler-response', 'Error: ' + error.message, true);
            }
        }
    </script>
</body>
</html>
            """
            return HTMLResponse(content=html_content)

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
                "is_sleeping": getattr(self, '_is_sleeping', False),
                "sleep_requested": getattr(self, '_sleep_requested', False),
                "current_buffer_size": len(self.buffer) if self.is_data_sampler else None,
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

        with self._generation_lock:
            # Check if sleep was requested while waiting for the lock
            if hasattr(self, '_sleep_requested') and self._sleep_requested:
                logger.info("Sleep requested - aborting buffer fill")
                return

            self._is_filling = True
            try:
                while len(self.buffer) < self.buffer_size:
                    # Check if sleep was requested during buffer fill
                    # if hasattr(self, '_sleep_requested') and self._sleep_requested:
                    #     logger.info("Sleep requested during buffer fill - stopping buffer fill")
                    #     break
                    
                    print(f"Buffer Size: {len(self.buffer)}")
                        
                    items = []
                    for _ in range(self._generation_batch_size):
                        try:
                            item = next(self.dataset_iter)
                            items.append(item)
                        except StopIteration:
                            self.dataset_iter = iter(self.dataset)
                            self._epoch += 1

                    start_time = time.perf_counter()
                    
                    generation_kwargs = {}
                    if self.config.single_gpu and os.path.exists(self.lora_path):
                        generation_kwargs["lora_request"] = LoRARequest("grpo", self._lora_idx, self.lora_path)
                        self._lora_idx += 1
                    # items_with_rewards = self.process_sample(items)
                    rollouts:list[Rollout] = []
                    for item in items:
                        r = Rollout(prompt=item[self.dataset_field], item=item)
                        rollouts.append(r)
                        
                    rollouts = self.environment.generate(rollouts, self.llm, self._sampling_params, vllm_generate_kwargs=generation_kwargs, tokenizer=self.tokenizer)
                    items_with_rewards = self.environment.payload(rollouts)
                    end_time = time.perf_counter()
                    print(f"time taken: {end_time - start_time}")
                    if len(items_with_rewards) == 0:
                        continue
                    print("==========")
                    for item in items_with_rewards:
                        print(f"{item['all_rewards']}")
                        print(f"{len(item['rewards'])} {item['rewards']}")
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

        # Check if sleep was requested
        if hasattr(self, '_sleep_requested') and self._sleep_requested:
            logger.info("Sleep requested - aborting sample processing")
            return []
            
        prompts = [item[self.dataset_field] for item in items]
        
        
        generation_kwargs = {}
        if self.config.single_gpu and os.path.exists(self.lora_path):
            generation_kwargs["lora_request"] = LoRARequest("grpo", self._lora_idx, self.lora_path)
            self._lora_idx += 1
        
        print(f">>>>> Generation_kwargs: {generation_kwargs} <<<<<")
        all_outputs = self.llm.generate(prompts, sampling_params=self._sampling_params, **generation_kwargs)
        
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
        if not self.is_data_sampler:
            return {}, [], 0.0, 0.0

        all_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function(
                self.tokenizer, items, completions, completion_ids
            )
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

        # Display information
        console.print("\n")
        console.print(Panel(table, title="[bold]Available Network Interfaces[/bold]"))
        console.print(Panel(status_text, title="[bold]Server Configuration[/bold]"))
        console.print(f"\n[bold blue]Server running on port {self.config.port}[/bold blue]\n")

        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
        dist.destroy_process_group()


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
