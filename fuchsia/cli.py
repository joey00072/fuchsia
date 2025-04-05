#!/usr/bin/env python3
import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the fuchsia command."""
    parser = argparse.ArgumentParser(
        description="Foosha - A collection of autoregressive model implementations and experiments"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run the vLLM server")
    server_parser.add_argument(
        "--model", required=True, help="Path to the model or model name"
    )
    server_parser.add_argument("--revision", help="Model revision to use")
    server_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    server_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server to"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    server_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    server_parser.add_argument(
        "--dtype",
        default="auto",
        help="Model data type (auto, float16, bfloat16, float32)",
    )
    server_parser.add_argument(
        "--max-model-len", type=int, default=512, help="Maximum model length"
    )
    server_parser.add_argument(
        "--enable-prefix-caching", action="store_true", help="Enable prefix caching"
    )
    server_parser.add_argument(
        "--quantization", help="Model quantization method (e.g., 'awq')"
    )

    # Add more subcommands here as needed

    args = parser.parse_args(argv)

    if args.command == "server":
        from .vllm_server import ServerConfig, VLLMServer

        config = ServerConfig(
            model=args.model,
            revision=args.revision,
            tensor_parallel_size=args.tensor_parallel_size,
            host=args.host,
            port=args.port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            enable_prefix_caching=args.enable_prefix_caching,
            quantization=args.quantization,
        )
        server = VLLMServer(config)
        server.serve()
        return 0
    elif args.command is None:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
