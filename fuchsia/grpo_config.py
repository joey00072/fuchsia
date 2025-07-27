import torch
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import yaml



@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training and generation.
    All parameters are user-configurable for maximum flexibility.
    """
    # Model configuration
    model_name: str = ""
    model_revision: Optional[str] = None
    max_model_len: int = 1024
    dtype: str = "bfloat16"
    
    
    group_size: int = 8
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    max_iterations: int = 1000
    log_wandb: bool = False
    lr: float = 5e-6
    weight_decay: float = 0.0
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    wandb_project: str = "fuchsia"
    dataset_feild: str = "prompt"
    num_policy_updates: int = 8
    using_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: Optional[List[str]] = None
    lora_path: str = "lora_weights"
    ignore_imcomplete_samples: bool = False
    use_clipping: str = "ppo"
    
    # Learning rate scheduler parameters
    use_scheduler: bool = True
    warmup_steps: int = 8
    scheduler_type: str = "constant_with_warmup"  # Options: "constant_with_warmup", "cosine", "linear"
    

    # Gradient checkpointing parameters
    gradient_checkpointing_enabled: bool = False
    gradient_checkpointing_cpu_offloading: bool = False
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.9
    repetition_penalty: float = 1.1
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0

    # Logging and saving
    log_level: str = "INFO"
    save_every: int = 25

    async_buffer_fill: bool = True
    debug: bool = True
    
    single_gpu: bool = False
    non_blocking: bool = False
    
    loss_type: str = "grpo"
    
    # Device
    device: Optional[str] = None  # If None, auto-detect

    def __post_init__(self):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        if self.dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {self.dtype}. Supported values are: {list(dtype_map.keys())}"
            )

        self.dtype = dtype_map[self.dtype]

        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.dtype = torch.float16

        # Set up logging
        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.INFO))
        self.logger = logging.getLogger("GRPO") 
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GRPOConfig":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Extract GRPO configuration
        grpo_config_dict = config.get("grpo", {})
        
        # Extract generation configuration for generation parameters
        generation_config = config.get("generation", {})
        
        # Extract model configuration
        model_config = config.get("model", {})
        
        # Extract dataset configuration
        dataset_config = config.get("dataset", {})
        
        # Extract training configuration
        training_config = config.get("training", {})
        
        # Extract server configuration (including nested vllm)
        server_config = config.get("server", {})
        vllm_config = server_config.get("vllm", {})
        
        # Extract LoRA configuration
        lora_config = config.get("lora", {})
        
        # Extract gradient checkpointing configuration
        gradient_checkpointing_config = grpo_config_dict.get("gradient_checkpointing", {})
        
        grpo_config = GRPOConfig(
            # GRPO specific parameters
            loss_type=grpo_config_dict.get("loss_type", "grpo"),
            group_size=grpo_config_dict.get("group_size", 8),
            batch_size=grpo_config_dict.get("batch_size", 1),
            grad_accumulation_steps=grpo_config_dict.get("grad_accumulation_steps", 1),
            lr=float(grpo_config_dict.get("lr", 5e-6)),
            weight_decay=float(grpo_config_dict.get("weight_decay", 0.0)),
            beta=float(grpo_config_dict.get("beta", 0.0)),
            epsilon=float(grpo_config_dict.get("epsilon", 0.2)),
            epsilon_high=float(grpo_config_dict.get("epsilon_high", 0.28)),
            log_wandb=grpo_config_dict.get("log_wandb", False),
            wandb_project=grpo_config_dict.get("wandb_project", "fuchsia"),
            num_policy_updates=grpo_config_dict.get("num_policy_updates", 8),
            lora_path=grpo_config_dict.get("lora_path", "lora_weights"),
            single_gpu=grpo_config_dict.get("single_gpu", False),
            
            # Learning rate scheduler parameters
            use_scheduler=grpo_config_dict.get("use_scheduler", True),
            warmup_steps=grpo_config_dict.get("warmup_steps", 8),
            scheduler_type=grpo_config_dict.get("scheduler_type", "constant_with_warmup"),
            
            # Gradient checkpointing configuration
            gradient_checkpointing_enabled=gradient_checkpointing_config.get("enabled", False),
            gradient_checkpointing_cpu_offloading=gradient_checkpointing_config.get("cpu_offloading", False),
            
            # Model configuration
            dtype=model_config.get("dtype", "bfloat16"),
            
            # Generation parameters with fallback to vllm config
            max_new_tokens=generation_config.get("max_len") or vllm_config.get("max_tokens", 512),
            temperature=float(generation_config.get("temperature") or vllm_config.get("temperature", 0.9)),
            top_p=float(generation_config.get("top_p") or vllm_config.get("top_p", 1.0)),
            top_k=int(generation_config.get("top_k") or vllm_config.get("top_k", -1)),
            min_p=float(generation_config.get("min_p") or vllm_config.get("min_p", 0.0)),
            
            # Dataset configuration
            dataset_feild=dataset_config.get("field", "prompt"),
            
            # Training configuration
            max_iterations=training_config.get("max_iterations", 1000),
            save_every=training_config.get("save_steps", 25),
            
            # LoRA configuration
            using_lora=lora_config.get("enabled", False),
            lora_r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 16),
            lora_target_modules=lora_config.get("target_modules", None),
            
            # Model configuration
            model_name=model_config.get("name", ""),
            model_revision=model_config.get("revision"),
            max_model_len=model_config.get("max_model_len", 1024),
        )

        return grpo_config