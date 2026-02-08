import torch
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import yaml
from fuchsia.rollout_queue import normalize_rollout_transfer_mode



@dataclass
class TrainerConfig:
    """
    Framework-level trainer configuration for rollout-based RL training.
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
    dataset_field: str = "prompt"
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

    # Sample transfer transport
    sample_transfer_mode: str = "api"  # Options: "api", "filesystem" ("http" alias supported)
    sample_transfer_dir: str = "/tmp/fuchsia_sample_queue"
    sample_transfer_poll_interval: float = 0.25

    # Importance sampling correction (off-policy stabilization)
    importance_sampling_correction: bool = True
    importance_ratio_type: str = "token"  # Options: "token", "sequence"
    importance_token_mask_high: float = 8.0
    importance_token_mask_low: float = 0.125
    importance_sequence_clip_high: float = 10.0
    importance_geo_mask_high: float = 10.0
    importance_geo_mask_low: float = 0.1
    importance_sequence_mask_low: float = 0.0
    importance_sequence_mask_high: float = 100.0
    
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

        if self.importance_ratio_type not in {"token", "sequence"}:
            raise ValueError(
                f"Unsupported importance_ratio_type: {self.importance_ratio_type}. "
                "Supported values are: ['token', 'sequence']"
            )
        if self.importance_token_mask_low >= self.importance_token_mask_high:
            raise ValueError(
                "importance_token_mask_low must be smaller than importance_token_mask_high"
            )
        if self.importance_geo_mask_low >= self.importance_geo_mask_high:
            raise ValueError(
                "importance_geo_mask_low must be smaller than importance_geo_mask_high"
            )
        if self.importance_sequence_mask_low >= self.importance_sequence_mask_high:
            raise ValueError(
                "importance_sequence_mask_low must be smaller than importance_sequence_mask_high"
            )
        self.sample_transfer_mode = normalize_rollout_transfer_mode(self.sample_transfer_mode)
        if self.sample_transfer_poll_interval <= 0:
            raise ValueError("sample_transfer_poll_interval must be > 0")

        # Set up logging
        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.INFO))
        self.logger = logging.getLogger("Trainer") 
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainerConfig":
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Extract trainer configuration
        trainer_config_dict = config.get("trainer", {})
        
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
        transfer_config = server_config.get("transfer", {})
        
        # Extract LoRA configuration
        lora_config = config.get("lora", {})
        
        # Extract gradient checkpointing configuration
        gradient_checkpointing_config = trainer_config_dict.get("gradient_checkpointing", {})
        importance_sampling_config = trainer_config_dict.get("importance_sampling", {})
        
        trainer_config = TrainerConfig(
            # Trainer-specific parameters (loss can be grpo/cispo/cxpo/reinforce etc.)
            loss_type=trainer_config_dict.get("loss_type", "grpo"),
            group_size=trainer_config_dict.get("group_size", 8),
            batch_size=trainer_config_dict.get("batch_size", 1),
            grad_accumulation_steps=trainer_config_dict.get("grad_accumulation_steps", 1),
            lr=float(trainer_config_dict.get("lr", 5e-6)),
            weight_decay=float(trainer_config_dict.get("weight_decay", 0.0)),
            beta=float(trainer_config_dict.get("beta", 0.0)),
            epsilon=float(trainer_config_dict.get("epsilon", 0.2)),
            epsilon_high=float(trainer_config_dict.get("epsilon_high", 0.28)),
            importance_sampling_correction=importance_sampling_config.get(
                "enabled", trainer_config_dict.get("importance_sampling_correction", True)
            ),
            importance_ratio_type=importance_sampling_config.get(
                "ratio_type", trainer_config_dict.get("importance_ratio_type", "token")
            ),
            importance_token_mask_high=float(
                importance_sampling_config.get(
                    "token_mask_high", trainer_config_dict.get("importance_token_mask_high", 8.0)
                )
            ),
            importance_token_mask_low=float(
                importance_sampling_config.get(
                    "token_mask_low", trainer_config_dict.get("importance_token_mask_low", 0.125)
                )
            ),
            importance_sequence_clip_high=float(
                importance_sampling_config.get(
                    "sequence_clip_high", trainer_config_dict.get("importance_sequence_clip_high", 10.0)
                )
            ),
            importance_geo_mask_high=float(
                importance_sampling_config.get(
                    "geo_mask_high", trainer_config_dict.get("importance_geo_mask_high", 10.0)
                )
            ),
            importance_geo_mask_low=float(
                importance_sampling_config.get(
                    "geo_mask_low", trainer_config_dict.get("importance_geo_mask_low", 0.1)
                )
            ),
            importance_sequence_mask_low=float(
                importance_sampling_config.get(
                    "sequence_mask_low", trainer_config_dict.get("importance_sequence_mask_low", 0.0)
                )
            ),
            importance_sequence_mask_high=float(
                importance_sampling_config.get(
                    "sequence_mask_high", trainer_config_dict.get("importance_sequence_mask_high", 100.0)
                )
            ),
            log_wandb=trainer_config_dict.get("log_wandb", False),
            wandb_project=trainer_config_dict.get("wandb_project", "fuchsia"),
            num_policy_updates=trainer_config_dict.get("num_policy_updates", 8),
            lora_path=trainer_config_dict.get("lora_path", "lora_weights"),
            single_gpu=trainer_config_dict.get("single_gpu", False),
            
            # Learning rate scheduler parameters
            use_scheduler=trainer_config_dict.get("use_scheduler", True),
            warmup_steps=trainer_config_dict.get("warmup_steps", 8),
            scheduler_type=trainer_config_dict.get("scheduler_type", "constant_with_warmup"),
            
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
            dataset_field=dataset_config.get("field", "prompt"),

            # Transfer configuration
            sample_transfer_mode=transfer_config.get(
                "mode", server_config.get("sample_transfer_mode", "api")
            ),
            sample_transfer_dir=transfer_config.get(
                "queue_dir",
                server_config.get("sample_transfer_dir", "/tmp/fuchsia_sample_queue"),
            ),
            sample_transfer_poll_interval=float(
                transfer_config.get(
                    "poll_interval",
                    server_config.get("sample_transfer_poll_interval", 0.25),
                )
            ),
            
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

        return trainer_config
