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
    run       Run server + trainer together (control plane)

Examples:
    # Create a default config file
    fuchsia config

    # Create a config file at a specific location
    fuchsia config --output custom_config.yaml

    # Run the vLLM server
    fuchsia server --model path/to/model

    # Run with a server script (trainer is auto-wired)
    fuchsia run examples/gsm8k/gsm8k_server.py
"""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from difflib import get_close_matches
import urllib.error
import urllib.request
import yaml
from pathlib import Path
from typing import Optional


RUN_PRESETS = {
    "gsm8k": "examples/gsm8k/gsm8k_server.py",
    "function": "examples/function/fn_server.py",
    "nanor1": "examples/nanor1/nanor1_server.py",
}


def _stream_process_output(process: subprocess.Popen, name: str) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[{name}] {line}", end="")


def _terminate_process(process: Optional[subprocess.Popen], name: str, timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {name} process (pid={process.pid})")
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{name} did not exit in {timeout:.1f}s, killing it")
        process.kill()
        process.wait()


def _wait_for_server_health(
    health_url: str,
    timeout_seconds: float,
    poll_interval: float,
    server_process: subprocess.Popen,
) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server_process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(poll_interval)
    return False


def _parse_extra_args(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    return shlex.split(raw)


def _resolve_server_script(args: argparse.Namespace) -> Path:
    server_script = None
    if args.example:
        server_script = RUN_PRESETS[args.example]
    elif args.server_script is not None:
        server_script = args.server_script
    elif args.server_script_flag is not None:
        server_script = args.server_script_flag

    if not server_script:
        raise ValueError(
            "Missing server script. Use `fuchsia run <server.py>` or `--example`."
        )

    server_candidate = Path(server_script).expanduser()
    if server_candidate.exists():
        return server_candidate.resolve()

    server_text = str(server_script)
    if "/" not in server_text and "\\" not in server_text:
        matches = sorted(Path("examples").rglob(server_text))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            options = ", ".join(str(path) for path in matches)
            raise ValueError(
                f"Ambiguous server script name '{server_script}'. "
                f"Found multiple matches: {options}. Use an explicit path."
            )

        known = sorted(str(path.name) for path in Path("examples").rglob("*_server.py"))
        suggestions = get_close_matches(server_text, known, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise FileNotFoundError(f"Server script not found: {server_script}.{hint}")

    raise FileNotFoundError(f"Server script not found: {server_script}")


def _infer_config_path(server_path: Path) -> Path:
    directory = server_path.parent
    stem = server_path.stem
    if stem.endswith("_server"):
        stem = stem[: -len("_server")]

    candidates = [
        directory / f"{stem}_config.yaml",
        directory / "config.yaml",
        directory / f"{stem}.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    fallback = sorted(directory.glob("*config*.yaml"))
    if len(fallback) == 1:
        return fallback[0].resolve()
    if len(fallback) > 1:
        options = ", ".join(str(path) for path in fallback)
        raise ValueError(
            f"Could not infer config uniquely for {server_path}. "
            f"Found multiple config candidates: {options}. Use --config."
        )

    raise FileNotFoundError(
        f"Could not find config YAML next to server script {server_path}. "
        "Use --config to specify one."
    )


def run_control_plane(args: argparse.Namespace) -> int:
    server_path = _resolve_server_script(args)
    python_bin = args.python

    server_cmd = [python_bin, str(server_path), *_parse_extra_args(args.server_args)]

    if args.trainer_script:
        trainer_path = Path(args.trainer_script).expanduser().resolve()
        if not trainer_path.exists():
            raise FileNotFoundError(f"Trainer script not found: {trainer_path}")
        trainer_cmd = [python_bin, str(trainer_path), *_parse_extra_args(args.trainer_args)]
    else:
        config_path = Path(args.config).expanduser().resolve() if args.config else _infer_config_path(server_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        trainer_cmd = [
            python_bin,
            "-m",
            "fuchsia.train",
            "--config",
            str(config_path),
            *_parse_extra_args(args.trainer_args),
        ]

    print(f"Starting server: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(
        server_cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    threading.Thread(
        target=_stream_process_output,
        args=(server_process, "server"),
        daemon=True,
    ).start()

    print(
        f"Waiting for server health endpoint {args.health_url} "
        f"(timeout={args.server_start_timeout:.1f}s)"
    )
    healthy = _wait_for_server_health(
        args.health_url,
        args.server_start_timeout,
        args.poll_interval,
        server_process,
    )
    if not healthy:
        print("Server did not become healthy before timeout, aborting run.")
        _terminate_process(server_process, "server")
        return 1

    print(f"Starting trainer: {' '.join(trainer_cmd)}")
    trainer_process = subprocess.Popen(
        trainer_cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    threading.Thread(
        target=_stream_process_output,
        args=(trainer_process, "trainer"),
        daemon=True,
    ).start()

    try:
        while True:
            trainer_rc = trainer_process.poll()
            server_rc = server_process.poll()

            if trainer_rc is not None:
                if trainer_rc == 0:
                    print("Trainer exited successfully; stopping server.")
                else:
                    print(f"Trainer exited with code {trainer_rc}; stopping server.")
                _terminate_process(server_process, "server")
                return trainer_rc

            if server_rc is not None:
                print(f"Server exited with code {server_rc}; stopping trainer.")
                _terminate_process(trainer_process, "trainer")
                return server_rc

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted. Stopping trainer and server.")
        _terminate_process(trainer_process, "trainer")
        _terminate_process(server_process, "server")
        return 130


def create_default_config(output_path: str = "config.yaml") -> None:
    """
    Create a default configuration file with predefined values.

    This function generates a YAML configuration file with default settings for Fuchsia.
    The configuration includes settings for generation, model, LoRA, trainer settings,
    server, dataset, and training parameters.

    Args:
        output_path (str): Path where the configuration file will be created.
                          Defaults to "config.yaml" in the current directory.

    The generated config file includes:
        - Generation parameters (max_len, temperature, etc.)
        - Model configuration (name, dtype, etc.)
        - LoRA settings (enabled, r, alpha, etc.)
        - Trainer configuration
        - Server settings
        - Dataset configuration
        - Training parameters
    """
    # Create a custom YAML dumper that preserves anchors
    class AnchorDumper(yaml.SafeDumper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.anchors = {}

        def anchor_node(self, node):
            if node in self.anchors:
                return self.anchors[node]
            anchor = f"anchor_{len(self.anchors)}"
            self.anchors[node] = anchor
            return anchor

    # Define the config with anchors
    default_config = {
        "shared": {
            "rollout": {
                "group_size": 8
            },
            "sampling": {
                "max_tokens": 512,
                "repetition_penalty": 1.0,
                "temperature": 0.6,
                "top_k": -1,
                "top_p": 1.0,
                "min_p": 0.0,
                "logprobs": 1
            },
            "runtime": {
                "single_gpu": False,
                "lora_path": "lora_weights"
            },
            "transfer": {
                "mode": "api",
                "queue_dir": "/tmp/fuchsia_sample_queue",
                "poll_interval": 0.25,
                "clear_on_start": False
            }
        },
        "generation": {
            "max_len": 512,
            "group_size": 8,
            "temperature": 0.6,
            "top_k": -1,
            "top_p": 1.0,
            "min_p": 0.0,
            "batch_size": 4
        },
        "model": {
            "name": "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2",
            "revision": None,
            "dtype": "bfloat16",
            "max_model_len": 512
        },
        "lora": {
            "enabled": True,
            "r": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        },
        "trainer": {
            "group_size": 8,
            "micro_group_size": 1,
            "batch_size": 1,
            "lr": 0.000005,
            "weight_decay": 0.1,
            "beta": 0.0,
            "epsilon": 0.2,
            "importance_sampling": {
                "enabled": True,
                "ratio_type": "token",
                "token_mask_high": 8.0,
                "token_mask_low": 0.125,
                "sequence_clip_high": 10.0,
                "geo_mask_high": 10.0,
                "geo_mask_low": 0.1,
                "sequence_mask_low": 0.0,
                "sequence_mask_high": 100.0
            },
            "log_wandb": True,
            "wandb_project": "fuchsia-jee-deephermes",
            "num_policy_updates": 8
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "gpu_memory_utilization": 0.50,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": False,
            "buffer_size": 4,
            "generation_batch_size": 4,
            "quantization": None,
            "transfer": {
                "mode": "api",
                "queue_dir": "/tmp/fuchsia_sample_queue",
                "poll_interval": 0.25,
                "clear_on_start": False
            },
            "vllm": {
                "max_tokens": 512,
                "n": 8,
                "repetition_penalty": 1.0,
                "temperature": 0.6,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "logprobs": 1
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

    # Create a YAML document with anchors
    yaml_doc = """# Shared configuration (used by both trainer and server)
shared:
  rollout:
    group_size: &group_size 8
  sampling:
    max_tokens: &max_tokens 512
    repetition_penalty: &repetition_penalty 1.0
    temperature: &temperature 0.6
    top_k: &top_k -1
    top_p: &top_p 1.0
    min_p: &min_p 0.0
    logprobs: &logprobs 1
  runtime:
    single_gpu: false
    lora_path: "lora_weights"
  transfer:
    mode: &transfer_mode "api"
    queue_dir: &transfer_queue_dir "/tmp/fuchsia_sample_queue"
    poll_interval: &transfer_poll_interval 0.25
    clear_on_start: &transfer_clear_on_start false

# Generation configuration (trainer-facing compatibility)
generation: &generation
  max_len: *max_tokens
  group_size: *group_size
  temperature: *temperature
  top_k: *top_k
  top_p: *top_p
  min_p: *min_p
  batch_size: &generation_batch_size 4

# Model configuration
model:
  name: "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"
  revision: null
  dtype: "bfloat16"
  max_model_len: *max_tokens

# LoRA configuration
lora:
  enabled: true
  r: 8
  alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

# Trainer configuration
trainer:
  group_size: *group_size
  micro_group_size: 1
  batch_size: 1
  lr: 0.000005
  weight_decay: 0.1
  beta: 0.0
  epsilon: 0.2
  importance_sampling:
    enabled: true
    ratio_type: "token"
    token_mask_high: 8.0
    token_mask_low: 0.125
    sequence_clip_high: 10.0
    geo_mask_high: 10.0
    geo_mask_low: 0.1
    sequence_mask_low: 0.0
    sequence_mask_high: 100.0
  log_wandb: true
  wandb_project: "fuchsia-jee-deephermes"
  num_policy_updates: 8

# Server configuration
server:
  host: "0.0.0.0"
  port: 8000
  gpu_memory_utilization: 0.50
  tensor_parallel_size: 1
  enable_prefix_caching: false
  buffer_size: 4
  generation_batch_size: *generation_batch_size
  quantization: null
  transfer:
    mode: *transfer_mode
    queue_dir: *transfer_queue_dir
    poll_interval: *transfer_poll_interval
    clear_on_start: *transfer_clear_on_start
  vllm:
    max_tokens: *max_tokens
    n: *group_size
    repetition_penalty: *repetition_penalty
    temperature: *temperature
    top_p: *top_p
    top_k: *top_k
    min_p: *min_p
    logprobs: *logprobs

# Dataset configuration
dataset:
  name: "AthenaAgent42/jee_papers"
  split: "train"
  max_samples: null
  field: "text"

# Training configuration
training:
  max_epochs: 1
  max_iterations: 1000
  save_steps: 100
  eval_steps: 50
  output_dir: "jee_output"
"""

    # Write the YAML document to file
    with open(output_path, "w") as f:
        f.write(yaml_doc)

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

    # Run with a server script (trainer auto-starts)
    fuchsia run examples/gsm8k/gsm8k_server.py
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

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run server and trainer together",
        description=(
            "Start a server script, wait for health, then start trainer. "
            "By default trainer is `python -m fuchsia.train --config <auto-detected-config>`."
        ),
    )
    run_parser.add_argument(
        "server_script",
        nargs="?",
        help="Path to server Python script. Example: examples/gsm8k/gsm8k_server.py",
    )
    run_parser.add_argument(
        "--example",
        choices=sorted(RUN_PRESETS.keys()),
        help="Use a built-in server preset (gsm8k, function, nanor1)",
    )
    run_parser.add_argument(
        "--server-script",
        dest="server_script_flag",
        help="Alias for positional server script path",
    )
    run_parser.add_argument(
        "--config",
        help="Path to trainer config YAML. If omitted, inferred from server script directory.",
    )
    run_parser.add_argument(
        "--trainer-script",
        help="Optional custom trainer script override",
    )
    run_parser.add_argument(
        "--python",
        default=sys.executable,
        help=f"Python executable to use (default: {sys.executable})",
    )
    run_parser.add_argument(
        "--server-args",
        default=None,
        help='Extra args for server script as one quoted string. Example: --server-args "--foo 1"',
    )
    run_parser.add_argument(
        "--trainer-args",
        default=None,
        help='Extra args for trainer process as one quoted string. Example: --trainer-args "--bar 2"',
    )
    run_parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:8000/health/",
        help="Health endpoint to wait for before starting trainer",
    )
    run_parser.add_argument(
        "--server-start-timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for server health",
    )
    run_parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Health polling interval in seconds",
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
    elif args.command == "run":
        try:
            return run_control_plane(args)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            return 1
    elif args.command is None:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
