
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

from datasets import Dataset,load_dataset
from transformers import AutoTokenizer

import asyncio
# Check CUDA availability
try:
    ctypes.CDLL("libcuda.so.1")
    libcuda_available = True
except OSError:
    libcuda_available = False

if libcuda_available:
    from vllm import LLM, SamplingParams
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

def get_ip_addresses():
    """Get all network IP addresses with detailed information"""
    ip_info = []
    
    # Get hostname
    hostname = socket.gethostname()
    ip_info.append({
        "interface": "Hostname",
        "ip": hostname,
        "type": "System"
    })
    
    # Get all network interfaces
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4 addresses
                ip_info.append({
                    "interface": interface,
                    "ip": addr.address,
                    "type": "IPv4",
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast if addr.broadcast else "N/A"
                })
            elif addr.family == socket.AF_INET6:  # IPv6 addresses
                ip_info.append({
                    "interface": interface,
                    "ip": addr.address,
                    "type": "IPv6",
                    "netmask": addr.netmask,
                    "broadcast": "N/A"
                })
    
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
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
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
            raise RuntimeError("Communicator not initialized")

        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
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
    def __init__(self, *args, **kwargs):
        known_fields = {
            'model', 'revision', 'tensor_parallel_size', 'host', 'port',
            'gpu_memory_utilization', 'dtype', 'max_model_len', 'enable_prefix_caching', 'quantization' 
        }
        
        for field in known_fields:
            if field in kwargs:
                setattr(self, field, kwargs.pop(field))
        
        for key, value in kwargs.items():
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


class VLLMServer:
    def __init__(self, config: ServerConfig):
        if not libcuda_available:
            raise ImportError("CUDA is required to run the vLLM serve script")

        self.config = config
        self.llm = LLM(
            model=config.model,
            revision=config.revision,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            enable_prefix_caching=config.enable_prefix_caching,
            max_model_len=config.max_model_len,
            worker_cls="fuchsia.vllm_server.WeightSyncWorker",
        )
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health/")
        async def health():
            """
            Health check endpoint to verify that the server is running.
            """
            return {"status": "ok"}

        @app.get("/get_tensor_parallel_size/")
        async def get_tensor_parallel_size():
            """
            Retrieves the tensor parallel size from the LLM engine.

            Returns:
                `dict`:
                    A dictionary containing the tensor parallel size.

            Example response:
            ```json
            {"tensor_parallel_size": 8}
            ```
            """
            return {"tensor_parallel_size": self.llm.llm_engine.parallel_config.tensor_parallel_size}

        @app.post("/generate/", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """
            Generates completions for the provided prompts.

            Args:
                request (`GenerateRequest`):
                    - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

            Returns:
                `GenerateResponse`:
                    - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

            Example request:
            ```json
            {"prompts": ["Hello world", "What is AI?"]}
            ```

            Example response:
            ```json
            {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
            ```
            """
            guided_decoding = None
            if request.guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)

            sampling_params = SamplingParams(
                n=request.n,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                max_tokens=request.max_tokens,
                guided_decoding=guided_decoding,
            )
            all_outputs = self.llm.generate(request.prompts, sampling_params=sampling_params)
            completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
            return {"completion_ids": completion_ids}

        @app.post("/init_communicator/")
        async def init_communicator(request: InitCommunicatorRequest, background_tasks: BackgroundTasks):
            """
            Initializes the communicator for synchronizing model weights between a client and multiple server
            workers.

            Args:
                request (`InitCommunicatorRequest`):
                    - `host` (`str`): Hostname or IP address of the master node.
                    - `port` (`int`): Port number to be used for communication.
                    - `world_size` (`int`): Total number of participating processes in the group.
            """
            background_tasks.add_task(
                self.llm.collective_rpc,
                "init_communicator",
                args=(request.host, request.port, self.config.tensor_parallel_size + 1),
            )
            return {"message": "Request received, initializing communicator"}

        @app.post("/update_named_param/")
        async def update_named_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
            """
            Updates the model weights with the provided tensor.

            Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

            Args:
                request (`UpdateWeightsRequest`):
                    - `name` (`str`): Name of the weight tensor being updated.
                    - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                    - `shape` (list of `int`): Shape of the weight
            """
            dtype = torch.__getattribute__(request.dtype.split(".")[-1])
            background_tasks.add_task(
                self.llm.collective_rpc, 
                "update_named_param", 
                args=(request.name, dtype, request.shape)
            )
            return {"message": "Request received, updating named parameter"}

        @app.post("/reset_prefix_cache/")
        async def reset_prefix_cache():
            """
            Resets the prefix cache for the model.
            """
            success = self.llm.llm_engine.reset_prefix_cache()
            return {"message": f"Reset prefix cache status: {success}"}

        @app.post("/close_communicator/")
        async def close_communicator():
            """
            Closes the weight update group and cleans up associated resources.
            """
            self.llm.collective_rpc("close_communicator")
            return {"message": "Request received, closing communicator"}

        return app

    def serve(self):
        """
        Starts the FastAPI server with the configured host and port.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        
        console = Console()
        
        # Create a table for IP addresses
        table = Table(title="Server Network Information", show_header=True, header_style="bold magenta")
        table.add_column("Interface", style="cyan")
        table.add_column("URL", style="green")
        
        # Filter and add only important interfaces (IPv4, excluding loopback)
        for ip_info in get_ip_addresses():
            if ip_info['type'] == 'IPv4' and ip_info['interface'] != 'lo':
                url = f"http://{ip_info['ip']}:{self.config.port}"
                table.add_row(
                    ip_info['interface'],
                    url
                )
        
        # Create a panel for the server status
        status_text = Text()
        status_text.append("Server Status: ", style="bold")
        status_text.append("Starting...", style="green")
        
        # Display the information
        console.print("\n")
        console.print(Panel(table, title="[bold]Available Network Interfaces[/bold]"))
        console.print(Panel(status_text, title="[bold]Server Configuration[/bold]"))
        console.print(f"\n[bold blue]Server running on port {self.config.port}[/bold blue]\n")
        
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
        dist.destroy_process_group()



class DataSamplerConfig(ServerConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.dataset_feild = kwargs.get("dataset_feild", "text")
        self.buffer_size = kwargs.get("buffer_size", 32)
        
        self.vllm_n:int = kwargs.get("vllm_n", 1)
        self.vllm_repetition_penalty:float = kwargs.get("vllm_repetition_penalty", 1.0)
        self.vllm_temperature:float = kwargs.get("vllm_temperature", 1.0)
        self.vllm_top_p:float = kwargs.get("vllm_top_p", 1.0)
        self.vllm_top_k:int = kwargs.get("vllm_top_k", -1)
        self.vllm_min_p:float = kwargs.get("vllm_min_p", 0.0)
        self.vllm_max_tokens:int = kwargs.get("vllm_max_tokens", 16)
        self.guided_decoding:Optional[GuidedDecodingParams] = kwargs.get("guided_decoding", None)

class DataSamplerServer(VLLMServer):
    def __init__(self, config: DataSamplerConfig, dataset: Dataset, reward_functions: list[Callable] = None):
        self.config = config    
        
        self.config = config
        self.llm = LLM(
            model=config.model,
            quantization=config.quantization,
            revision=config.revision,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            enable_prefix_caching=config.enable_prefix_caching,
            max_model_len=config.max_model_len,
            worker_cls="fuchsia.vllm_server.WeightSyncWorker",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)    
        self.dataset = dataset
        self.dataset_feild = config.dataset_feild
        self.reward_functions = reward_functions
        self.buffer_size = config.buffer_size
        self.buffer = []
        self.dataset_iter = iter(self.dataset)
        self._epoch = 1  
        self._generation_batch_size = 1
        self._sampling_params = SamplingParams(
            n=config.vllm_n,
            repetition_penalty=config.vllm_repetition_penalty,
            temperature=config.vllm_temperature,
            top_p=config.vllm_top_p,
            top_k=config.vllm_top_k,
            min_p=config.vllm_min_p,
            max_tokens=config.vllm_max_tokens,
            guided_decoding=config.guided_decoding,
        )   
        
        self._is_filling = False  # Add this line to track fill status
        self.app = self._create_app()
        
    def _create_app(self) -> FastAPI:
        app = super()._create_app()
        
        self.buffer_fill()
        
        
        @app.post("/get_sample/")
        async def get_sample(background_tasks: BackgroundTasks):
            """
            Returns a sample from the buffer and triggers background buffer fill.
            """
            if len(self.buffer) == 0:
                
                await asyncio.sleep(5)
                return {"sample": None}
            sample = self.buffer.pop(0)
            # Add buffer fill task to run in background after response
            background_tasks.add_task(self.buffer_fill)
            
            return {"sample": sample}
        
        @app.post("/buffer_fill/")
        async def buffer_fill(background_tasks: BackgroundTasks):
            """
            Fills the buffer with new samples if not already filling.
            """
            if self._is_filling:
                return {"message": "Buffer fill already in progress"}
            
            background_tasks.add_task(self.buffer_fill)
            return {"message": "Buffer filling started"}
        return app
    
    def buffer_fill(self):
        if self._is_filling:
            return
            
        self._is_filling = True
        try:
            while len(self.buffer) < self.buffer_size:
                samples = []
                for _ in range(self._generation_batch_size):        
                    try:
                        sample = next(self.dataset_iter)
                        samples.append(sample)
                    except StopIteration:
                        self.dataset_iter = iter(self.dataset)
                        self._epoch += 1
                
                samples_with_rewards = self.process_sample(samples)
                self.buffer.append(samples_with_rewards)
        finally:
            self._is_filling = False
    
    def process_sample(self, samples):
        prompts = [sample[self.dataset_feild] for sample in samples]
        guided_decoding = None
        sampling_params = SamplingParams(
            n=self.config.vllm_n,
            repetition_penalty=self.config.vllm_repetition_penalty,
            temperature=self.config.vllm_temperature,
            top_p=self.config.vllm_top_p,
            top_k=self.config.vllm_top_k,
            min_p=self.config.vllm_min_p,
            max_tokens=self.config.vllm_max_tokens,
            guided_decoding=guided_decoding,
        )
        all_outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        
        completions = [self.tokenizer.decode(c) for c in completion_ids]
        rewards = self.calculate_rewards(samples, completions)
        
        return {"completion_ids": completion_ids, "completions": completions, "rewards": rewards}
        
    def calculate_rewards(self, samples, completions):
        rewards = [[] for _ in range(len(self.reward_functions))]
        for sample, completion in zip(samples, completions):
            for idx, reward_function in enumerate(self.reward_functions):
                rewards[idx].append(reward_function(sample, completion))
        return rewards
        

def load_config_from_yaml(yaml_path: str) -> ServerConfig:
    """
    Load server configuration from a YAML file.
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        ServerConfig: Configuration object populated from YAML
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ServerConfig(**config_dict)

def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to load (e.g., HuggingFace model ID or local path)")
    parser.add_argument("--revision", type=str)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max_model_len", default=1024, type=int)
    parser.add_argument("--enable_prefix_caching", default=False, type=bool)
    parser.add_argument("--config", type=str, default="examples/vllm_server_config.yaml", 
                       help="Path to YAML configuration file (default: examples/vllm_server_config.yaml)")

    args = parser.parse_args()
    
    # If config file is provided, load from YAML
    if args.config:
        config = load_config_from_yaml(args.config)
        # Override with any CLI arguments that were provided
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                setattr(config, key, value)
    else:
        config = ServerConfig(**vars(args))
    
    # Ensure model is specified either through config or CLI
    if not config.model:
        parser.error("Model must be specified either through --model argument or in the config file")
    
    server = VLLMServer(config)
    server.serve()


def test_datasampler():
    config = DataSamplerConfig(
        model="unsloth/Llama-3.2-3B-Instruct",
        revision="main",
        tensor_parallel_size=1,
        host="0.0.0.0",
        port=8000,
        dataset_feild="Question Text",
        buffer_size=4,
        max_model_len=1024,
        gpu_memory_utilization=0.7,
        dtype="bfloat16",
        quantization="fp8",
    )
    ds = load_dataset("CK0607/2025-Jee-Mains-Question",split="train")
    def reward_function(samples, completions):
        return [len(completion) for completion in completions]
    
    server = DataSamplerServer(config, ds, [reward_function])
    server.serve()  
    
if __name__ == "__main__":
    # main()
    test_datasampler()