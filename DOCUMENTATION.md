# Fuchsia Framework - Comprehensive API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration System](#configuration-system)
4. [Core Components](#core-components)
5. [VLLM Integration](#vllm-integration)
6. [Environment System](#environment-system)
7. [Utilities](#utilities)
8. [Command Line Interface](#command-line-interface)
9. [Examples and Usage Patterns](#examples-and-usage-patterns)
10. [API Reference](#api-reference)

---

## Overview

Fuchsia is a flexible and efficient framework for Group Relative Policy Optimization (GRPO) training with support for LoRA fine-tuning, distributed training, and various model configurations. It provides a comprehensive suite of tools for reinforcement learning on Large Language Models (LLMs).

### Key Features

- **GRPO Training**: Implementation of Group Relative Policy Optimization algorithm
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **Distributed Training**: Support for multi-GPU setups with VLLM integration
- **Single GPU Training**: Works on single GPU with hotswap mode
- **Gradient Checkpointing**: Memory-efficient training options with CPU offloading
- **Flexible Environment System**: Support for single-turn and multi-turn environments
- **Comprehensive Configuration**: YAML-based configuration system

---

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-compatible GPU
- PyTorch >= 2.0.0

### Install from Source

```bash
git clone https://github.com/joey00072/fuchsia
cd fuchsia
pip install -e .
```

### Dependencies

Core dependencies include:
- `torch>=2.0.0`
- `transformers>=4.36.0` 
- `vllm==0.8.1`
- `peft>=0.7.0`
- `datasets>=2.15.0`
- `wandb>=0.16.0`

See `pyproject.toml` for the complete list.

---

## Configuration System

### GRPOConfig Class

The central configuration class that controls all aspects of GRPO training.

#### Creating a Configuration

```python
from fuchsia.grpo_config import GRPOConfig

# From YAML file
config = GRPOConfig.from_yaml("config.yaml")

# Programmatically
config = GRPOConfig(
    model_name="microsoft/DialoGPT-medium",
    group_size=8,
    batch_size=2,
    lr=5e-6,
    using_lora=True,
    lora_r=8
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `""` | HuggingFace model name or path |
| `group_size` | `int` | `8` | Number of completions per prompt |
| `batch_size` | `int` | `1` | Training batch size |
| `lr` | `float` | `5e-6` | Learning rate |
| `using_lora` | `bool` | `False` | Enable LoRA fine-tuning |
| `lora_r` | `int` | `8` | LoRA rank |
| `lora_alpha` | `int` | `16` | LoRA alpha parameter |
| `epsilon` | `float` | `0.2` | PPO clipping parameter |
| `beta` | `float` | `0.0` | KL divergence penalty weight |
| `max_iterations` | `int` | `1000` | Maximum training iterations |
| `log_wandb` | `bool` | `False` | Enable W&B logging |

#### YAML Configuration Example

```yaml
# Generation configuration
generation: &generation
  max_len: &max_len 512
  group_size: &group_size 8
  temperature: &temperature 0.6
  batch_size: &generation_batch_size 4

# Model configuration  
model:
  name: "microsoft/DialoGPT-medium"
  dtype: "bfloat16"
  max_model_len: *max_len

# LoRA configuration
lora:
  enabled: true
  r: 8
  alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# GRPO training configuration
grpo:
  group_size: *group_size
  batch_size: 1
  lr: 0.000005
  epsilon: 0.2
  log_wandb: true
  wandb_project: "fuchsia-training"
```

---

## Core Components

### GRPO Class

The main training class implementing Group Relative Policy Optimization.

#### Initialization

```python
from fuchsia.grpo import GRPO
from fuchsia.grpo_config import GRPOConfig
from fuchsia.vllm_client import VLLMClient

config = GRPOConfig.from_yaml("config.yaml")
vllm_client = VLLMClient()

grpo = GRPO(
    model=model,              # PreTrainedModel or model name
    ref_model=ref_model,      # Optional reference model
    tokenizer=tokenizer,      # PreTrainedTokenizer
    dataset=dataset,          # Dataset iterator
    optimizer=optimizer,      # Optional custom optimizer
    config=config,            # GRPOConfig instance
    vllm_client=vllm_client  # VLLMClient instance
)
```

#### Key Methods

##### `train(epochs=1, max_iterations=10000)`

Starts the GRPO training loop.

```python
# Basic training
grpo.train(max_iterations=1000)

# Multi-epoch training
grpo.train(epochs=3, max_iterations=5000)
```

##### `compute_loss(inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask, ignore_sample)`

Computes the GRPO loss with optional KL divergence penalty.

**Parameters:**
- `inputs`: Input token sequences
- `old_policy_log_probs`: Log probabilities from previous policy
- `reward`: Reward values for each completion
- `mean_rewards`: Mean reward for normalization
- `std_rewards`: Standard deviation for normalization
- `loss_mask`: Mask for valid tokens
- `ignore_sample`: Mask for incomplete samples

**Returns:**
- `loss`: Computed loss tensor
- `avg_kld`: Average KL divergence

##### Memory Management Methods

```python
# CPU offloading for memory efficiency
grpo.offload_to_cpu()

# Load model back to GPU
grpo.load_model_to_gpu()

# Clean GPU memory
grpo.clean_and_sync_memory()
```

##### Policy Update Handling

```python
# Update VLLM server with new model weights
grpo.handle_policy_update()
```

#### Training Loop Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from fuchsia.grpo import GRPO
from fuchsia.grpo_config import GRPOConfig
from fuchsia.vllm_client import VLLMClient
from fuchsia.dist_dataset import DatasetClient

# Configuration
config = GRPOConfig.from_yaml("config.yaml")

# Model setup
model = AutoModelForCausalLM.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

if config.using_lora:
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules
    )
    model = get_peft_model(model, lora_config)

# Dataset and VLLM client
vllm_client = VLLMClient()
dataset = DatasetClient(vllm_client)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.lr, 
    weight_decay=config.weight_decay
)

# Initialize trainer
grpo = GRPO(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    optimizer=optimizer,
    config=config,
    vllm_client=vllm_client
)

# Start training
grpo.train(max_iterations=config.max_iterations)
```

---

## VLLM Integration

### VLLMServer Class

High-performance server for model inference using VLLM.

#### ServerConfig

Configuration for the VLLM server.

```python
from fuchsia.vllm_server import ServerConfig

config = ServerConfig(
    model="microsoft/DialoGPT-medium",
    host="0.0.0.0",
    port=8000,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    dtype="bfloat16",
    max_model_len=512
)
```

#### Starting the Server

```python
from fuchsia.vllm_server import VLLMServer

server = VLLMServer(config)
server.serve()  # Starts the server
```

#### DataSamplerServer

Extended server with data sampling capabilities.

```python
from fuchsia.vllm_server import DataSamplerServer
from datasets import load_dataset

dataset = load_dataset("your-dataset")
reward_functions = [your_reward_function]

server = DataSamplerServer(
    config=config,
    dataset=dataset,
    reward_functions=reward_functions,
    pre_fill_buffer=True
)

server.serve()
```

#### API Endpoints

The server exposes several REST endpoints:

- `GET /health/` - Health check
- `POST /generate/` - Generate completions
- `POST /sleep/` - Put server to sleep (memory optimization)
- `POST /wake_up/` - Wake up server
- `POST /get_sample/` - Get processed sample (DataSampler mode)
- `POST /buffer_fill/` - Fill generation buffer

### VLLMClient Class

Client for communicating with VLLM server.

#### Initialization

```python
from fuchsia.vllm_client import VLLMClient

client = VLLMClient(
    host="0.0.0.0",
    server_port=8000,
    group_port=51216,
    init_communicator=True
)
```

#### Key Methods

##### `generate(prompts, **kwargs)`

Generate completions for given prompts.

```python
prompts = ["Hello, how are you?", "What is AI?"]
completions = client.generate(
    prompts=prompts,
    n=4,                    # Number of completions per prompt
    temperature=0.8,
    max_tokens=100,
    top_p=0.9
)
```

##### `update_model_params(model, lora=False, single_gpu=False)`

Update server model parameters after training.

```python
# Update full model parameters
client.update_model_params(model, lora=False)

# Update LoRA parameters only
client.update_model_params(model, lora=True)

# Single GPU mode (saves to disk)
client.update_model_params(model, single_gpu=True, lora_path="./lora_weights")
```

##### Sleep/Wake Methods

```python
# Put server to sleep (memory optimization)
client.sleep()

# Wake up server
client.wake_up()
```

##### Buffer Management

```python
# Fill generation buffer
client.fill_buffer(num_samples=100)

# Empty buffer
client.empty_buffer()

# Get sample from buffer
sample = client.get_sample()
```

#### Fault Tolerance

All client methods include built-in retry logic and graceful error handling:

```python
# Client automatically retries failed requests
try:
    completions = client.generate(prompts)
except Exception as e:
    # Client will retry 3 times before raising exception
    print(f"Generation failed after retries: {e}")
```

---

## Environment System

### Base Environment Class

Framework for defining RL environments.

```python
from fuchsia.envs import Environment
from vllm import SamplingParams

env = Environment(
    reward_functions=[reward_fn1, reward_fn2],
    sampling_params=SamplingParams(temperature=0.8),
    max_samples=5,
    stop=["</answer>"]
)
```

### Rollout Class

Represents a single trajectory through an environment.

```python
from fuchsia.envs import Rollout

rollout = Rollout(
    prompt="Solve this problem:",
    completion="",
    state={"step": 0},
    item={"problem_id": 123}
)
```

#### Rollout Properties

- `prompt`: Initial prompt text
- `completion`: Generated completion text  
- `completion_ids`: Token IDs of completion
- `finish_reason`: Why generation stopped ("stop", "length", etc.)
- `completed`: Whether rollout is complete
- `rewards`: Dictionary of reward values
- `mean`/`std`: Reward statistics

### SingleTurnEnvironment

For single-step interactions.

```python
from fuchsia.envs import SingleTurnEnvironment

def accuracy_reward(rollouts, items, completions, completion_ids):
    """Reward function that checks answer accuracy."""
    rewards = []
    for item, completion in zip(items, completions):
        correct = check_answer(item['question'], completion)
        rewards.append(1.0 if correct else 0.0)
    return rewards

env = SingleTurnEnvironment(
    reward_functions=[accuracy_reward],
    max_samples=1
)

# Generate rollouts
rollouts = env.generate(initial_rollouts, llm, sampling_params)
```

### MultiTurnEnvironment

For multi-step interactions with environment feedback.

```python
from fuchsia.envs import MultiTurnEnvironment

class MathEnvironment(MultiTurnEnvironment):
    def step_rollout(self, rollout):
        """Process rollout and provide feedback."""
        if "<calculate>" in rollout.last_completion:
            # Extract calculation and add result
            calc = extract_calculation(rollout.last_completion)
            result = evaluate(calc)
            rollout.completion += f"\nResult: {result}"
        return rollout

env = MathEnvironment(
    reward_functions=[math_reward],
    max_samples=3,
    stop=["</solution>"]
)
```

### PythonEnvironment

Pre-built environment for Python code execution.

```python
from fuchsia.envs import PythonEnvironment

def code_reward(rollouts, items, completions, completion_ids):
    """Reward based on code execution success."""
    rewards = []
    for rollout in rollouts:
        if "Error:" not in rollout.completion:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

env = PythonEnvironment()
env.reward_functions = [code_reward]
```

#### Environment Usage Example

```python
from fuchsia.envs import SingleTurnEnvironment, Rollout
from vllm import LLM, SamplingParams

# Define reward function
def simple_reward(rollouts, items, completions, completion_ids):
    return [len(comp.split()) / 10.0 for comp in completions]  # Reward based on length

# Create environment
env = SingleTurnEnvironment(
    reward_functions=[simple_reward],
    max_samples=1
)

# Create initial rollouts
rollouts = [
    Rollout(prompt="Tell me about AI:", item={"id": 1}),
    Rollout(prompt="What is Python?", item={"id": 2})
]

# Generate completions
llm = LLM(model="gpt2")
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

completed_rollouts = env.generate(rollouts, llm, sampling_params)

# Get training data
samples = env.payload(completed_rollouts, calculate_rewards=True)
```

---

## Utilities

### Network Utilities

#### `get_ip_addresses()`

Get detailed network interface information.

```python
from fuchsia.utils import get_ip_addresses

ip_info = get_ip_addresses()
for info in ip_info:
    print(f"{info['interface']}: {info['ip']} ({info['type']})")
```

### GPU Information

#### `gpu_info(model)`

Get comprehensive GPU and model information.

```python
from fuchsia.utils import gpu_info

info = gpu_info(model)
print(f"GPU: {info['device_name']}")
print(f"Memory: {info['memory_allocated_mb']:.1f}MB / {info['memory_reserved_mb']:.1f}MB")
print(f"Rank: {info['rank']}/{info['world_size']}")
```

### CPU Offloading

#### Gradient Checkpointing with CPU Offloading

```python
from fuchsia.cpu_offloding import apply_cpu_gradient_checkpoint_monkey_patch

# Apply monkey patch for memory-efficient training
apply_cpu_gradient_checkpoint_monkey_patch()

# Enable gradient checkpointing on model
model.gradient_checkpointing_enable()
```

### Dataset Client

#### DatasetClient

Client for consuming samples from VLLM server.

```python
from fuchsia.dist_dataset import DatasetClient
from fuchsia.vllm_client import VLLMClient

client = VLLMClient()
dataset = DatasetClient(client)

# Iterate over samples
for sample in dataset:
    print(f"Completions: {sample['completions']}")
    print(f"Rewards: {sample['rewards']}")
    break
```

---

## Command Line Interface

### Available Commands

#### `fuchsia config`

Create default configuration file.

```bash
# Create config.yaml in current directory
fuchsia config

# Create config at specific location
fuchsia config --output my_config.yaml
```

#### `fuchsia server`

Start VLLM server.

```bash
# Basic server
fuchsia server --model microsoft/DialoGPT-medium

# Advanced configuration
fuchsia server \
  --model microsoft/DialoGPT-medium \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --port 8001 \
  --dtype bfloat16 \
  --enable-prefix-caching
```

#### Server Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name or path (required) | - |
| `--revision` | Model revision | `None` |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism | `1` |
| `--host` | Host to bind server | `0.0.0.0` |
| `--port` | Port to bind server | `8000` |
| `--gpu-memory-utilization` | GPU memory usage (0.0-1.0) | `0.5` |
| `--dtype` | Model data type | `auto` |
| `--max-model-len` | Maximum model length | `512` |
| `--enable-prefix-caching` | Enable prefix caching | `False` |
| `--quantization` | Quantization method | `None` |

### Configuration Generation

The CLI can generate comprehensive configuration files:

```bash
fuchsia config --output complete_config.yaml
```

This creates a YAML file with all configuration options and anchors for easy customization.

---

## Examples and Usage Patterns

### Basic Training Setup

```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from fuchsia.grpo import GRPO
from fuchsia.grpo_config import GRPOConfig
from fuchsia.vllm_client import VLLMClient
from fuchsia.dist_dataset import DatasetClient

def setup_training():
    # Load configuration
    config = GRPOConfig.from_yaml("config.yaml")
    
    # Initialize VLLM client
    vllm_client = VLLMClient(init_communicator=False)
    vllm_client.sleep()
    
    # Setup dataset
    dataset = DatasetClient(vllm_client)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="cpu",
        torch_dtype=config.dtype,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Setup LoRA if enabled
    if config.using_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules
        )
        model = get_peft_model(model, lora_config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    return model, tokenizer, dataset, optimizer, config, vllm_client

def main():
    model, tokenizer, dataset, optimizer, config, vllm_client = setup_training()
    
    # Initialize GRPO trainer
    grpo = GRPO(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        config=config,
        vllm_client=vllm_client
    )
    
    # Start training
    grpo.train(max_iterations=config.max_iterations)

if __name__ == "__main__":
    main()
```

### Custom Reward Function

```python
def mathematical_accuracy_reward(rollouts, items, completions, completion_ids):
    """
    Reward function for mathematical problem solving.
    
    Args:
        rollouts: List of Rollout objects
        items: List of dataset items
        completions: List of completion strings
        completion_ids: List of token ID lists
        
    Returns:
        List of reward values
    """
    rewards = []
    
    for item, completion in zip(items, completions):
        # Extract expected answer
        expected = item.get('answer', '')
        
        # Extract model's answer (assuming format: "Answer: X")
        import re
        match = re.search(r'Answer:\s*([+-]?\d*\.?\d+)', completion)
        if match:
            try:
                predicted = float(match.group(1))
                expected_num = float(expected)
                
                # Exact match gets full reward
                if abs(predicted - expected_num) < 1e-6:
                    reward = 1.0
                # Close match gets partial reward  
                elif abs(predicted - expected_num) / abs(expected_num) < 0.1:
                    reward = 0.5
                else:
                    reward = 0.0
            except ValueError:
                reward = 0.0
        else:
            reward = 0.0  # No answer found
            
        rewards.append(reward)
    
    return rewards
```

### Multi-GPU Setup

```python
# config.yaml for multi-GPU
"""
server:
  tensor_parallel_size: 4  # Use 4 GPUs
  gpu_memory_utilization: 0.8
  
grpo:
  single_gpu: false  # Disable single GPU mode
  batch_size: 8      # Larger batch size for multi-GPU
"""

# Training script remains the same
# The framework automatically handles multi-GPU communication
```

### Memory-Efficient Training

```python
from fuchsia.cpu_offloding import apply_cpu_gradient_checkpoint_monkey_patch

def setup_memory_efficient_training():
    # Apply CPU offloading patch
    apply_cpu_gradient_checkpoint_monkey_patch()
    
    config = GRPOConfig.from_yaml("config.yaml")
    
    # Enable gradient checkpointing
    config.gradient_checkpointing_enabled = True
    config.gradient_checkpointing_cpu_offloading = True
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="cpu",  # Start on CPU
        torch_dtype=torch.bfloat16,
        use_cache=False    # Required for gradient checkpointing
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    return model, config
```

### Server with Custom Dataset

```python
from datasets import load_dataset
from fuchsia.vllm_server import ServerConfig, DataSamplerServer

def custom_reward_function(rollouts, items, completions, completion_ids):
    # Your custom reward logic
    return [1.0 if "good" in comp else 0.0 for comp in completions]

# Load dataset
dataset = load_dataset("your-dataset-name")["train"]

# Server configuration  
config = ServerConfig.from_yaml("server_config.yaml")

# Start server with dataset
server = DataSamplerServer(
    config=config,
    dataset=dataset,
    reward_functions=[custom_reward_function],
    pre_fill_buffer=True
)

server.serve()
```

---

## API Reference

### Complete Class and Function Reference

#### fuchsia.grpo

##### GRPO

```python
class GRPO:
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Optional[PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        dataset: Iterator[Dict[str, Any]],
        optimizer: Optional[torch.optim.Optimizer] = None,
        reward_functions: Optional[List[Callable]] = None,
        config: GRPOConfig = None,
        vllm_client: Optional[VLLMClient] = None,
    )
    
    def train(self, epochs: int = 1, max_iterations: int = 10000) -> None
    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask, ignore_sample) -> Tuple[Tensor, float]
    def sample_batch(self) -> Tuple[Tensor, ...]
    def offload_to_cpu(self) -> PreTrainedModel
    def load_model_to_gpu(self) -> PreTrainedModel
    def handle_policy_update(self) -> None
    def log_metrics(self) -> None
```

#### fuchsia.grpo_config

##### GRPOConfig

```python
@dataclass
class GRPOConfig:
    # Model configuration
    model_name: str = ""
    model_revision: Optional[str] = None
    max_model_len: int = 1024
    dtype: str = "bfloat16"
    
    # Training configuration
    group_size: int = 8
    batch_size: int = 1
    lr: float = 5e-6
    weight_decay: float = 0.0
    beta: float = 0.0
    epsilon: float = 0.2
    max_iterations: int = 1000
    
    # LoRA configuration
    using_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: Optional[List[str]] = None
    
    # Logging and saving
    log_wandb: bool = False
    wandb_project: str = "fuchsia"
    save_every: int = 25
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GRPOConfig"
```

#### fuchsia.vllm_server

##### ServerConfig

```python
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
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ServerConfig"
```

##### DataSamplerServer

```python
class DataSamplerServer:
    def __init__(
        self,
        config: ServerConfig,
        dataset: Optional[Dataset] = None,
        reward_functions: Optional[list[Callable]] = None,
        pre_fill_buffer: bool = True,
        environment: Environment = None,
        stop: Optional[list[str]] = None,
    )
    
    def serve(self) -> None
    def buffer_fill(self) -> None
```

#### fuchsia.vllm_client

##### VLLMClient

```python
class VLLMClient:
    def __init__(
        self,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
        init_communicator: bool = True,
    )
    
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 16,
        **kwargs
    ) -> list[list[str]]
    
    def update_model_params(self, model: nn.Module, lora=False, single_gpu=False, lora_path=None) -> None
    def sleep(self) -> dict
    def wake_up(self) -> dict
    def fill_buffer(self, num_samples: int = None) -> dict
    def empty_buffer(self) -> dict
    def get_sample(self) -> Optional[dict]
```

#### fuchsia.envs

##### Environment

```python
@dataclass
class Environment:
    reward_functions: list[Callable] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    max_samples: int = 1
    stop: list[str] = field(default_factory=list)
    
    def generate(self, rollouts, llm, sampling_params=None, **kwargs) -> list[Rollout]
    def payload(self, rollouts, calculate_rewards=True) -> list[dict]
    def calculate_rewards(self, items, completions, completion_ids, rollouts) -> tuple
```

##### Rollout

```python
@dataclass
class Rollout:
    prompt: str = ""
    completion: str = ""
    completion_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""
    completed: bool = False
    rewards: dict = field(default_factory=dict)
    mean: float = 0.0
    std: float = 0.0
    
    @property
    def is_completed(self) -> bool
    @property  
    def input(self) -> str
    def clone(self) -> "Rollout"
```

#### fuchsia.cli

##### Main CLI Functions

```python
def create_default_config(output_path: str = "config.yaml") -> None
def main(argv: Optional[list[str]] = None) -> int
```

#### fuchsia.utils

##### Utility Functions

```python
def get_ip_addresses() -> list[dict]
def gpu_info(model: torch.nn.Module) -> dict[str, Any]
```

#### fuchsia.cpu_offloding

##### Memory Optimization

```python
def apply_cpu_gradient_checkpoint_monkey_patch() -> None

class CPUGradientCheckpointer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward_fn, activations, *kwargs)
    @staticmethod 
    def backward(ctx, grad_output)
```

---

## Best Practices

### Configuration Management

1. **Use YAML configs**: Keep all hyperparameters in YAML files for reproducibility
2. **Use anchors**: Leverage YAML anchors to avoid duplication
3. **Environment-specific configs**: Maintain separate configs for different environments

### Memory Management

1. **Enable gradient checkpointing**: For large models, use gradient checkpointing with CPU offloading
2. **Single GPU mode**: Use single GPU mode with hotswap for memory-constrained setups
3. **Monitor memory**: Use `gpu_info()` to track memory usage

### Training Best Practices

1. **Start small**: Begin with smaller models and datasets to validate setup
2. **Log everything**: Enable W&B logging to track training progress
3. **Validate rewards**: Test reward functions independently before training
4. **Checkpoint frequently**: Save model checkpoints regularly

### Production Deployment

1. **Use multi-GPU**: Scale to multiple GPUs for production workloads
2. **Monitor health**: Implement health checks for long-running servers
3. **Handle failures**: Leverage built-in retry mechanisms in client code

---

This documentation covers all major public APIs, functions, and components in the Fuchsia framework. For specific use cases or advanced configurations, refer to the examples directory and configuration files in the repository.