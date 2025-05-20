#!/usr/bin/env python3
"""
Fuchsia CLI - Command Line Interface for Fuchsia

This module provides the command-line interface for Fuchsia, a collection of autoregressive model
implementations and experiments. It includes commands for configuration management, server operations,
and other utilities.

Usage:
    fuchsia [command] [options]

Commands:
    config    Create or manage configuration files
    server    Run the vLLM server for model serving

Examples:
    # Create a default config file
    fuchsia config

    # Create a config file at a specific location
    fuchsia config --output custom_config.yaml

    # Run the vLLM server
    fuchsia server --model path/to/model
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional


def get_default_config_dict() -> dict:
    """
    Returns the default configuration dictionary for Fuchsia.

    This dictionary includes settings for generation, model, LoRA, GRPO training,
    server, dataset, and training parameters. Shared values are defined as common
    Python objects to enable YAML anchor/alias generation if dumped.
    """
    # Define shared values as common Python objects to enable YAML anchors/aliases
    max_len_val = 512
    group_size_val = 8
    temperature_val = 0.6
    top_k_val = -1
    top_p_val = 1.0
    min_p_val = 0.0
    generation_batch_size_val = 4

    return {
        "generation": {
            "max_len": max_len_val,
            "group_size": group_size_val,
            "temperature": temperature_val,
            "top_k": top_k_val,
            "top_p": top_p_val,
            "min_p": min_p_val,
            "batch_size": generation_batch_size_val
        },
        "model": {
            "name": "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2",
            "revision": None,
            "dtype": "bfloat16",
            "max_model_len": max_len_val  # Reference to shared object
        },
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        },
        "grpo": {
            "group_size": group_size_val,  # Reference to shared object
            "micro_group_size": 1,
            "batch_size": 1,
            "lr": 0.000005,
            "weight_decay": 0.1,
            "beta": 0.0,
            "epsilon": 0.2,
            "log_wandb": True,
            "wandb_project": "fuchsia-jee-deephermes",
            "num_policy_updates": 8,
            "lora_path": "/mnt/nvme0n1/joey/experiments/lora_weights3"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "gpu_memory_utilization": 0.50,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": False,
            "buffer_size": 4,
            "generation_batch_size": generation_batch_size_val,  # Reference to shared object
            "quantization": None,
            "vllm": {
                "max_tokens": max_len_val,       # Reference
                "n": group_size_val,            # Reference
                "temperature": temperature_val, # Reference
                "top_p": top_p_val,             # Reference
                "top_k": top_k_val,             # Reference
                "min_p": min_p_val              # Reference
            }
        },
        "dataset": {
            "name": "AthenaAgent42/jee_papers",
            "split": "train",
            "max_samples": None,
            "field": "text"
        },
        "training": {
            "max_epochs": 1,
            "max_iterations": 1000,
            "save_steps": 100,
            "eval_steps": 50,
            "output_dir": "jee_output"
        }
    }


def create_default_config(output_path: str = "config.yaml") -> None:
    """
    Create a default configuration file with predefined values.

    This function generates a YAML configuration file with default settings for Fuchsia.
    The configuration includes settings for generation, model, LoRA, GRPO training,
    server, dataset, and training parameters.

    Args:
        output_path (str): Path where the configuration file will be created.
                          Defaults to "config.yaml" in the current directory.

    The generated config file includes:
        - Generation parameters (max_len, temperature, etc.)
        - Model configuration (name, dtype, etc.)
        - LoRA settings (enabled, r, alpha, etc.)
        - GRPO training configuration
        - Server settings
        - Dataset configuration
        - Training parameters
    """
    # Create a custom YAML dumper.
    # Note: This AnchorDumper's anchor_node method is not automatically used by yaml.dump
    # in a way that replicates the original hand-crafted YAML's specific anchor names
    # or comments. PyYAML will generate its own anchors if objects are reused.
    # Comments and specific anchor names like '&generation:' from the original yaml_doc
    # will be lost.
    class AnchorDumper(yaml.SafeDumper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The self.anchors dictionary here was likely for a custom representer setup,
            # which is not fully implemented for this use case.
            # We'll leave the class structure as is, but it won't replicate original comments/anchor names.

    default_config = get_default_config_dict()

    # Write the config dictionary to file using yaml.dump
    # Using sort_keys=False to preserve insertion order as much as possible.
    # The AnchorDumper class is passed, but standard object reuse in default_config
    # is what will trigger anchor/alias creation by PyYAML.
    # Comments and specific anchor names from the original yaml_doc cannot be
    # easily replicated with this approach.
    with open(output_path, "w") as f:
        yaml.dump(default_config, f, Dumper=AnchorDumper, sort_keys=False, indent=2, default_flow_style=None)

    print(f"Created default config file at {output_path}")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main entry point for the Fuchsia command-line interface.

    This function sets up the argument parser and handles command dispatching.
    It supports various subcommands for different Fuchsia operations.

    Args:
        argv (Optional[list[str]]): Command line arguments. If None, uses sys.argv[1:].

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Fuchsia - A collection of autoregressive model implementations and experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a default config file
    fuchsia config

    # Create a config file at a specific location
    fuchsia config --output custom_config.yaml

    # Run the vLLM server
    fuchsia server --model path/to/model
"""
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Create or manage configuration files",
        description="Create a default configuration file with predefined settings."
    )
    config_parser.add_argument(
        "--output", 
        default="config.yaml",
        help="Output path for the config file (default: config.yaml)"
    )

    # Server command
    server_parser = subparsers.add_parser(
        "server",
        help="Run the vLLM server",
        description="Start the vLLM server for model serving."
    )
    server_parser.add_argument(
        "--model",
        required=True,
        help="Path to the model or model name (required)"
    )
    server_parser.add_argument(
        "--revision",
        help="Model revision to use"
    )
    server_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    server_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization (0.0 to 1.0, default: 0.5)"
    )
    server_parser.add_argument(
        "--dtype",
        default="auto",
        help="Model data type (auto, float16, bfloat16, float32, default: auto)"
    )
    server_parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model length (default: 512)"
    )
    server_parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Enable prefix caching"
    )
    server_parser.add_argument(
        "--quantization",
        help="Model quantization method (e.g., 'awq')"
    )

    args = parser.parse_args(argv)

    if args.command == "config":
        create_default_config(args.output)
        return 0
    elif args.command == "server":
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
