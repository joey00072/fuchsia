"""Distributed vLLM server with multi-GPU and multi-node support.

This module is adapted from the upstream TRL implementation to provide
multi-GPU and multi-node generation capabilities within the Fuchsia
project.
"""

import argparse
import base64
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import torch
from transformers import is_vision_available

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)


if is_fastapi_available():
    from fastapi import FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vision_available():
    from PIL import Image


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """Worker extension enabling weight synchronisation across processes."""

    pynccl_comm = None
    client_rank = None

    def init_communicator(self, host: str, port: int, world_size: int, client_device_uuid: str) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        if client_device_uuid == str(torch.cuda.get_device_properties(self.device).uuid):
            raise RuntimeError(
                "Attempting to use the same CUDA device for multiple distinct roles. Ensure that the trainer "
                "uses different devices than the vLLM server."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None



@dataclass
class ScriptArguments:
    r"""Arguments for the distributed server."""

    model: str = field(metadata={"help": "Model name or path."})
    revision: Optional[str] = field(
        default=None, metadata={"help": "Revision to use for the model."}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of tensor parallel workers."}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Number of data parallel workers."}
    )
    host: str = field(default="0.0.0.0", metadata={"help": "Host address."})
    port: int = field(default=8000, metadata={"help": "Port to bind the server."})
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "Fraction of GPU memory to reserve for model and cache."},
    )
    dtype: str = field(
        default="auto",
        metadata={"help": "Data type for vLLM generation."},
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Optional max model length for vLLM."},
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable prefix caching."},
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={"help": "Force eager execution."},
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={"help": "Data type for KV cache."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading models."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Log level for uvicorn."},
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={"help": "Model implementation to use in vLLM."},
    )


def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="fuchsia.vllm_dp_server.WeightSyncWorkerExtension",
        trust_remote_code=script_args.trust_remote_code,
        model_impl=script_args.vllm_model_impl,
    )

    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break

        if command["type"] in ["call", "fire_and_forget"]:
            method = getattr(llm, command["method"])
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list[list]:
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]



def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError(
            "vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`."
        )

    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(
            target=llm_worker,
            args=(script_args, data_parallel_rank, master_port, child_connection),
        )
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)
        yield
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(
                    f"Process {process} is still alive after 10 seconds, attempting to terminate..."
                )
                process.terminate()
                process.join()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {
            "world_size": script_args.tensor_parallel_size * script_args.data_parallel_size
        }

    class GenerateRequest(BaseModel):
        prompts: list[str]
        images: Optional[list[str]] = None
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        generation_kwargs: dict = field(default_factory=dict)

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        request.images = request.images or [None] * len(request.prompts)
        prompts = []
        for prompt, image in zip(request.prompts, request.images):
            row = {"prompt": prompt}
            if image is not None:
                row["multi_modal_data"] = {"image": Image.open(BytesIO(base64.b64decode(image)))}
            prompts.append(row)

        guided_decoding = None
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "guided_decoding": guided_decoding,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        chunked_prompts = chunk_list(prompts, script_args.data_parallel_size)
        for connection, prompts in zip(connections, chunked_prompts):
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]
        all_outputs = list(chain.from_iterable(all_outputs))
        completion_ids = [
            list(output.token_ids) for outputs in all_outputs for output in outputs.outputs
        ]
        return {"completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        kwargs = {
            "method": "init_communicator",
            "args": (request.host, request.port, world_size, request.client_device_uuid),
        }
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        kwargs = {
            "method": "update_named_param",
            "args": (request.name, request.dtype, tuple(request.shape)),
        }
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve", help="Run the vLLM distributed serve script", dataclass_types=ScriptArguments
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)

