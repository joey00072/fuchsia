# Fuchsia - RL Training Framework

A flexible and efficient framework for rollout-based RL training with support for LoRA fine-tuning, distributed training, and interactive environments.

## Features

- **Trainer with Multiple Losses**: Supports `grpo`, `cispo`, `cxpo`, and `reinforce`
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **Distributed Training**: Support for multi-GPU setups with VLLM backend
- **Single GPU Training**: Works on single GPU with hotswap mode
- **Gradient Checkpointing**: Memory-efficient training with CPU offloading
- **Interactive Environments**: Support for tool-calling and multi-turn environments
- **Flexible Reward Functions**: Custom reward calculation for different tasks

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fuchsia

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Fuchsia uses a server-client architecture where a VLLM server generates completions and a training client performs policy updates.

### 1. Run Server + Trainer Together (Recommended)

Start both processes with one command:

```bash
# Pass only the server script
fuchsia run examples/gsm8k/gsm8k_server.py
fuchsia run examples/function/fn_server.py
```

This command starts the server first, waits for its health endpoint, auto-detects the config in that folder, and then starts the generic trainer.

### 2. Manual Two-Terminal Mode

If you want manual control, run the server and generic trainer separately:

```bash
# Terminal 1
python examples/gsm8k/gsm8k_server.py

# Terminal 2
python -m fuchsia.train --config examples/gsm8k/gsm8k_config.yaml
```

## Examples

### Function Calling with IPython

The `examples/function/` directory demonstrates training a model to use tools:

- **Server** (`fn_server.py`): Serves completions with tool-calling environment
- **Training** (`fuchsia.train`): Generic trainer module driven by config
- **Config** (`config.yaml`): Configuration for model, LoRA, and training parameters

**Key Features:**
- IPython interpreter tool integration
- Multi-turn conversations with tool calls
- Custom reward functions for format compliance and correctness
- MCP (Model Context Protocol) integration

### GSM8K Math Problem Solving

The `examples/gsm8k/` directory shows training on mathematical reasoning:

- **Server** (`gsm8k_server.py`): Serves math problems with thinking format
- **Training** (`fuchsia.train`): Generic trainer module driven by config
- **Config** (`gsm8k_config.yaml`): Optimized settings for math problems

**Key Features:**
- Structured thinking and answer format
- Reward functions for format and correctness
- GSM8K dataset integration
- Progressive reward scaling

## Configuration

Both examples use YAML configuration files with the following sections:

### Model Configuration
```yaml
model:
  name: "joey00072/exp-ntr-qwen3-4b-v0"  # HuggingFace model name
  dtype: "bfloat16"                       # Model precision
  max_model_len: 2048                     # Maximum sequence length
```

### LoRA Configuration
```yaml
lora:
  enabled: true
  r: 8                                    # LoRA rank
  alpha: 16                              # LoRA alpha scaling
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
```

### Shared Configuration
```yaml
shared:
  rollout:
    group_size: 8                       # Shared rollout count for trainer + server
  sampling:
    max_tokens: 2048
    temperature: 0.8
    top_p: 1.0
    top_k: -1
    min_p: 0.0
  runtime:
    single_gpu: true
    lora_path: "./lora_weights"
  transfer:
    mode: "api"                         # "api" or "filesystem"
    queue_dir: "/tmp/fuchsia_sample_queue"
```

### Trainer Configuration
```yaml
trainer:
  loss_type: "reinforce"                 # Loss type (grpo/cispo/cxpo/reinforce)
  batch_size: 8                          # Training batch size
  lr: 0.00005                           # Learning rate
  weight_decay: 0.2                     # Weight decay
  beta: 0.01                            # KL penalty coefficient
  num_policy_updates: 1                 # Updates per iteration
  gradient_checkpointing:
    enabled: true                        # Enable gradient checkpointing
    cpu_offloading: true                # Offload to CPU for memory savings
```

### Server Configuration
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  gpu_memory_utilization: 0.95          # GPU memory usage
  tensor_parallel_size: 1               # Multi-GPU parallelism
  quantization: fp8                     # Model quantization
```

## Workflow

1. **Configuration**: Set up your model, dataset, and training parameters in YAML
2. **Server Launch**: Start the VLLM server with your model configuration
3. **Training**: Run the training script which connects to the server
4. **Monitoring**: Track progress via Weights & Biases integration
5. **Evaluation**: Monitor rewards and model performance

## Architecture

- **VLLM Server**: High-performance inference server for generating completions
- **Trainer**: Handles policy optimization and LoRA updates
- **Environment**: Manages multi-turn interactions and reward calculation
- **Dataset Client**: Interfaces between training and server for data flow

## Advanced Usage

### Custom Environments

Create custom environments by inheriting from `MultiTurnEnvironment`:

```python
from fuchsia.envs import MultiTurnEnvironment, Rollout

class CustomEnvironment(MultiTurnEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = 4
        self.stop = ["</tool_call>"]
        self.reward_functions = [custom_reward_function]

    def step_rollout(self, rollout: Rollout):
        # Custom logic for processing each step
        pass
```

### Custom Reward Functions

Define custom reward functions for your specific task:

```python
def custom_reward_function(rollouts: list[Rollout], *args, **kwargs) -> list[float]:
    rewards = []
    for rollout in rollouts:
        # Calculate reward based on completion quality
        reward = calculate_reward(rollout.completion, rollout.item)
        rewards.append(reward)
    return rewards
```

### Memory Optimization

For training on limited GPU memory:

1. **Enable gradient checkpointing with CPU offloading**:
```yaml
trainer:
  gradient_checkpointing:
    enabled: true
    cpu_offloading: true
```

2. **Use quantization**:
```yaml
server:
  quantization: fp8  # or int8
```

3. **Reduce batch sizes**:
```yaml
trainer:
  batch_size: 2
  group_size: 4
```

## Troubleshooting

### Common Issues

1. **Server connection errors**: Ensure VLLM server is running before starting training
2. **GPU memory issues**: Reduce batch size, enable gradient checkpointing, or use quantization
3. **Model loading errors**: Verify model name and HuggingFace access tokens if needed

### Performance Tips

- Use `fp8` quantization for faster inference
- Enable prefix caching for repeated prompts
- Adjust `gpu_memory_utilization` based on your GPU
- Use larger `group_size` for better exploration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- VLLM
- PEFT
- Datasets
- Rich (for logging)
- Weights & Biases (optional, for logging)
- FastMCP (for tool integration)
