import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Literal
from enum import Enum
import yaml
import warnings



class SchedulerType(Enum):
    """Supported learning rate scheduler types."""
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    COSINE = "cosine"
    LINEAR = "linear"


class ClippingType(Enum):
    """Supported gradient clipping types."""
    PPO = "ppo"
    NONE = "none"


class LossType(Enum):
    """Supported loss types for training."""
    GRPO = "grpo"
    PPO = "ppo"
    DPO = "dpo"


@dataclass
class GRPOConfig:
    """
    Comprehensive configuration class for GRPO (Group Relative Policy Optimization) training and generation.
    
    This configuration class provides extensive customization options for:
    - Model settings and hardware configuration
    - Training hyperparameters and optimization settings
    - LoRA (Low-Rank Adaptation) configuration
    - Generation parameters for text completion
    - Logging, monitoring, and checkpointing
    - Learning rate scheduling and gradient management
    
    All parameters include detailed documentation, type hints, and validation to ensure
    robust configuration management and clear understanding of each option's purpose.
    """
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    model_name: str = field(
        default="",
        metadata={
            "description": "Name or path of the model to use for training and inference",
            "examples": ["microsoft/DialoGPT-medium", "gpt2", "./local/model/path"],
            "required": True,
            "category": "model"
        }
    )
    
    model_revision: Optional[str] = field(
        default=None,
        metadata={
            "description": "Specific model revision/commit hash to use from HuggingFace Hub",
            "examples": ["main", "v1.0", "abc123def456"],
            "category": "model"
        }
    )
    
    max_model_len: int = field(
        default=1024,
        metadata={
            "description": "Maximum sequence length the model can handle",
            "range": (1, 32768),
            "unit": "tokens",
            "category": "model",
            "impact": "Higher values increase memory usage but allow longer sequences"
        }
    )
    
    dtype: str = field(
        default="bfloat16",
        metadata={
            "description": "Data type for model weights and computations",
            "choices": ["bfloat16", "float16", "float32", "float64"],
            "category": "model",
            "recommendations": {
                "bfloat16": "Best for modern GPUs with BF16 support (A100, H100)",
                "float16": "Good for older GPUs, may have numerical instability",
                "float32": "Most stable but uses 2x memory",
                "float64": "Highest precision but uses 4x memory"
            }
        }
    )
    
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    
    group_size: int = field(
        default=8,
        metadata={
            "description": "Number of responses generated per prompt for group-based optimization",
            "range": (1, 64),
            "category": "training",
            "impact": "Higher values provide better gradient estimates but increase memory usage"
        }
    )
    
    batch_size: int = field(
        default=1,
        metadata={
            "description": "Number of training examples processed simultaneously",
            "range": (1, 128),
            "category": "training",
            "impact": "Higher values speed up training but require more memory"
        }
    )
    
    grad_accumulation_steps: int = field(
        default=1,
        metadata={
            "description": "Number of steps to accumulate gradients before updating weights",
            "range": (1, 1000),
            "category": "training",
            "usage": "Effective batch size = batch_size * grad_accumulation_steps"
        }
    )
    
    max_iterations: int = field(
        default=1000,
        metadata={
            "description": "Maximum number of training iterations to perform",
            "range": (1, 1000000),
            "category": "training"
        }
    )
    
    lr: float = field(
        default=5e-6,
        metadata={
            "description": "Learning rate for the optimizer",
            "range": (1e-8, 1e-2),
            "category": "training",
            "recommendations": {
                "small_models": "1e-5 to 5e-5",
                "large_models": "1e-6 to 1e-5",
                "fine_tuning": "1e-6 to 5e-6"
            }
        }
    )
    
    weight_decay: float = field(
        default=0.0,
        metadata={
            "description": "L2 regularization coefficient to prevent overfitting",
            "range": (0.0, 1.0),
            "category": "training",
            "typical_values": [0.0, 0.01, 0.1]
        }
    )
    
    beta: float = field(
        default=0.0,
        metadata={
            "description": "KL divergence coefficient for policy regularization",
            "range": (0.0, 1.0),
            "category": "training",
            "impact": "Higher values keep policy closer to reference model"
        }
    )
    
    epsilon: float = field(
        default=0.2,
        metadata={
            "description": "Clipping parameter for policy gradient methods",
            "range": (0.01, 1.0),
            "category": "training",
            "typical_values": [0.1, 0.2, 0.3]
        }
    )
    
    epsilon_high: float = field(
        default=0.28,
        metadata={
            "description": "Upper bound for adaptive clipping in GRPO",
            "range": (0.01, 1.0),
            "category": "training",
            "constraint": "Should be >= epsilon"
        }
    )
    
    num_policy_updates: int = field(
        default=8,
        metadata={
            "description": "Number of policy update steps per training iteration",
            "range": (1, 50),
            "category": "training"
        }
    )
    
    use_clipping: str = field(
        default="ppo",
        metadata={
            "description": "Type of gradient clipping to apply",
            "choices": ["ppo", "none"],
            "category": "training"
        }
    )
    
    loss_type: str = field(
        default="grpo",
        metadata={
            "description": "Type of loss function to use for training",
            "choices": ["grpo", "ppo", "dpo"],
            "category": "training",
            "recommendations": {
                "grpo": "Group-based optimization, best for multi-response scenarios",
                "ppo": "Standard PPO loss, good baseline",
                "dpo": "Direct preference optimization"
            }
        }
    )
    
    # ============================================================================
    # LEARNING RATE SCHEDULER
    # ============================================================================
    
    use_scheduler: bool = field(
        default=True,
        metadata={
            "description": "Whether to use learning rate scheduling",
            "category": "scheduler",
            "impact": "Scheduling can improve convergence and final performance"
        }
    )
    
    warmup_steps: int = field(
        default=8,
        metadata={
            "description": "Number of steps for learning rate warmup",
            "range": (0, 1000),
            "category": "scheduler",
            "usage": "Gradually increases LR from 0 to target LR over these steps"
        }
    )
    
    scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={
            "description": "Type of learning rate scheduler to use",
            "choices": ["constant_with_warmup", "cosine", "linear"],
            "category": "scheduler",
            "descriptions": {
                "constant_with_warmup": "Warmup then constant LR",
                "cosine": "Cosine annealing schedule",
                "linear": "Linear decay schedule"
            }
        }
    )
    

    # ============================================================================
    # GRADIENT CHECKPOINTING
    # ============================================================================
    
    gradient_checkpointing_enabled: bool = field(
        default=False,
        metadata={
            "description": "Enable gradient checkpointing to save memory",
            "category": "memory",
            "trade_off": "Reduces memory usage but increases computation time"
        }
    )
    
    gradient_checkpointing_cpu_offloading: bool = field(
        default=False,
        metadata={
            "description": "Offload checkpointed gradients to CPU memory",
            "category": "memory",
            "requires": "gradient_checkpointing_enabled=True",
            "impact": "Further reduces GPU memory but may slow training"
        }
    )
    
    # ============================================================================
    # TEXT GENERATION PARAMETERS
    # ============================================================================
    
    max_new_tokens: int = field(
        default=512,
        metadata={
            "description": "Maximum number of new tokens to generate per response",
            "range": (1, 4096),
            "category": "generation",
            "unit": "tokens"
        }
    )
    
    temperature: float = field(
        default=0.9,
        metadata={
            "description": "Sampling temperature for text generation",
            "range": (0.01, 2.0),
            "category": "generation",
            "impact": "Higher values increase randomness, lower values increase determinism",
            "typical_values": [0.7, 0.8, 0.9, 1.0, 1.1]
        }
    )
    
    repetition_penalty: float = field(
        default=1.1,
        metadata={
            "description": "Penalty for repeating tokens in generated text",
            "range": (0.5, 2.0),
            "category": "generation",
            "usage": "1.0 = no penalty, >1.0 = discourage repetition, <1.0 = encourage repetition"
        }
    )
    
    top_p: float = field(
        default=1.0,
        metadata={
            "description": "Nucleus sampling: cumulative probability threshold",
            "range": (0.0, 1.0),
            "category": "generation",
            "usage": "1.0 = disabled, 0.9 = use top 90% probability mass"
        }
    )
    
    top_k: int = field(
        default=-1,
        metadata={
            "description": "Top-k sampling: number of highest probability tokens to consider",
            "range": (-1, 1000),
            "category": "generation",
            "usage": "-1 = disabled, positive values limit vocabulary"
        }
    )
    
    min_p: float = field(
        default=0.0,
        metadata={
            "description": "Minimum probability threshold for token selection",
            "range": (0.0, 1.0),
            "category": "generation",
            "usage": "Filters out tokens below this probability"
        }
    )

    # ============================================================================
    # LORA (LOW-RANK ADAPTATION) CONFIGURATION
    # ============================================================================
    
    using_lora: bool = field(
        default=False,
        metadata={
            "description": "Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning",
            "category": "lora",
            "benefits": "Reduces trainable parameters and memory usage"
        }
    )
    
    lora_r: int = field(
        default=8,
        metadata={
            "description": "LoRA rank - dimensionality of adaptation matrices",
            "range": (1, 256),
            "category": "lora",
            "impact": "Higher values increase capacity but also parameter count",
            "typical_values": [4, 8, 16, 32, 64]
        }
    )
    
    lora_alpha: int = field(
        default=16,
        metadata={
            "description": "LoRA scaling parameter",
            "range": (1, 512),
            "category": "lora",
            "usage": "Controls the magnitude of LoRA updates (alpha/r ratio is important)",
            "typical_values": [8, 16, 32, 64]
        }
    )
    
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "description": "List of module names to apply LoRA to",
            "category": "lora",
            "examples": [["q_proj", "v_proj"], ["query", "key", "value"], ["all-linear"]],
            "usage": "None = auto-detect, list = specific modules"
        }
    )
    
    lora_path: str = field(
        default="lora_weights",
        metadata={
            "description": "Path to save/load LoRA weights",
            "category": "lora"
        }
    )
    
    # ============================================================================
    # DATASET CONFIGURATION
    # ============================================================================
    
    dataset_feild: str = field(
        default="prompt",
        metadata={
            "description": "Field name in dataset containing input prompts",
            "category": "dataset",
            "examples": ["prompt", "text", "input", "question"]
        }
    )
    
    ignore_imcomplete_samples: bool = field(
        default=False,
        metadata={
            "description": "Skip samples that don't meet completion criteria",
            "category": "dataset"
        }
    )
    
    # ============================================================================
    # LOGGING AND MONITORING
    # ============================================================================
    
    log_level: str = field(
        default="INFO",
        metadata={
            "description": "Logging verbosity level",
            "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "category": "logging"
        }
    )
    
    log_wandb: bool = field(
        default=False,
        metadata={
            "description": "Enable Weights & Biases logging for experiment tracking",
            "category": "logging",
            "requires": "wandb library installed"
        }
    )
    
    wandb_project: str = field(
        default="fuchsia",
        metadata={
            "description": "W&B project name for organizing experiments",
            "category": "logging"
        }
    )
    
    save_every: int = field(
        default=25,
        metadata={
            "description": "Save model checkpoint every N iterations",
            "range": (1, 1000),
            "category": "checkpointing",
            "unit": "iterations"
        }
    )
    
    # ============================================================================
    # SYSTEM AND PERFORMANCE
    # ============================================================================
    
    async_buffer_fill: bool = field(
        default=True,
        metadata={
            "description": "Fill data buffer asynchronously to improve throughput",
            "category": "performance"
        }
    )
    
    debug: bool = field(
        default=True,
        metadata={
            "description": "Enable debug mode with additional logging and checks",
            "category": "debug",
            "impact": "May slow down training but provides more information"
        }
    )
    
    single_gpu: bool = field(
        default=False,
        metadata={
            "description": "Force single GPU usage even in multi-GPU environments",
            "category": "hardware"
        }
    )
    
    non_blocking: bool = field(
        default=False,
        metadata={
            "description": "Use non-blocking CUDA operations for better performance",
            "category": "performance"
        }
    )
    
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    
    device: Optional[str] = field(
        default=None,
        metadata={
            "description": "Device to use for training and inference",
            "choices": ["auto", "cuda", "cpu", "cuda:0", "cuda:1"],
            "category": "hardware",
            "usage": "None/auto = automatic detection, specific device = force usage"
        }
    )

    def __post_init__(self):
        """
        Post-initialization validation and setup.
        
        Performs comprehensive validation of configuration parameters,
        sets up logging, handles device detection, and ensures compatibility
        between different configuration options.
        """
        self._validate_configuration()
        self._setup_dtype()
        self._setup_logging()
        self._validate_dependencies()
        
    def _validate_configuration(self):
        """Validate configuration parameters and ranges."""
        errors = []
        
        # Model validation
        if not self.model_name.strip():
            errors.append("model_name cannot be empty")
            
        if self.max_model_len <= 0:
            errors.append(f"max_model_len must be positive, got {self.max_model_len}")
            
        # Training parameter validation
        if self.group_size <= 0:
            errors.append(f"group_size must be positive, got {self.group_size}")
            
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
            
        if self.grad_accumulation_steps <= 0:
            errors.append(f"grad_accumulation_steps must be positive, got {self.grad_accumulation_steps}")
            
        if self.lr <= 0:
            errors.append(f"learning rate must be positive, got {self.lr}")
            
        if not 0 <= self.weight_decay <= 1:
            errors.append(f"weight_decay must be in [0, 1], got {self.weight_decay}")
            
        if not 0 <= self.beta <= 1:
            errors.append(f"beta must be in [0, 1], got {self.beta}")
            
        if not 0 < self.epsilon <= 1:
            errors.append(f"epsilon must be in (0, 1], got {self.epsilon}")
            
        if not 0 < self.epsilon_high <= 1:
            errors.append(f"epsilon_high must be in (0, 1], got {self.epsilon_high}")
            
        if self.epsilon_high < self.epsilon:
            errors.append(f"epsilon_high ({self.epsilon_high}) must be >= epsilon ({self.epsilon})")
            
        # Generation parameter validation
        if self.max_new_tokens <= 0:
            errors.append(f"max_new_tokens must be positive, got {self.max_new_tokens}")
            
        if not 0 < self.temperature <= 2.0:
            errors.append(f"temperature must be in (0, 2], got {self.temperature}")
            
        if not 0 <= self.top_p <= 1:
            errors.append(f"top_p must be in [0, 1], got {self.top_p}")
            
        if not 0 <= self.min_p <= 1:
            errors.append(f"min_p must be in [0, 1], got {self.min_p}")
            
        # LoRA validation
        if self.using_lora:
            if self.lora_r <= 0:
                errors.append(f"lora_r must be positive when using LoRA, got {self.lora_r}")
            if self.lora_alpha <= 0:
                errors.append(f"lora_alpha must be positive when using LoRA, got {self.lora_alpha}")
                
        # Scheduler validation
        valid_schedulers = ["constant_with_warmup", "cosine", "linear"]
        if self.scheduler_type not in valid_schedulers:
            errors.append(f"scheduler_type must be one of {valid_schedulers}, got '{self.scheduler_type}'")
            
        # Clipping validation
        valid_clipping = ["ppo", "none"]
        if self.use_clipping not in valid_clipping:
            errors.append(f"use_clipping must be one of {valid_clipping}, got '{self.use_clipping}'")
            
        # Loss type validation
        valid_loss_types = ["grpo", "ppo", "dpo"]
        if self.loss_type not in valid_loss_types:
            errors.append(f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'")
            
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def _setup_dtype(self):
        """Setup and validate dtype configuration."""
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

        original_dtype = self.dtype
        self.dtype = dtype_map[self.dtype]

        # Handle bfloat16 compatibility
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            warnings.warn(
                f"bfloat16 not supported on this device. Falling back to float16.",
                UserWarning
            )
            self.dtype = torch.float16
            
    def _setup_logging(self):
        """Setup logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Must be one of {valid_levels}")
            
        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.INFO))
        self.logger = logging.getLogger("GRPO")
        
    def _validate_dependencies(self):
        """Validate dependencies between configuration options."""
        # Gradient checkpointing dependencies
        if self.gradient_checkpointing_cpu_offloading and not self.gradient_checkpointing_enabled:
            warnings.warn(
                "gradient_checkpointing_cpu_offloading requires gradient_checkpointing_enabled=True",
                UserWarning
            )
            
        # W&B logging dependencies
        if self.log_wandb:
            try:
                import wandb
            except ImportError:
                warnings.warn(
                    "wandb library not installed but log_wandb=True. Install with: pip install wandb",
                    UserWarning
                )
                
    def get_effective_batch_size(self) -> int:
        """Calculate the effective batch size including gradient accumulation."""
        return self.batch_size * self.grad_accumulation_steps
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of key configuration parameters."""
        return {
            "model": {
                "name": self.model_name,
                "max_length": self.max_model_len,
                "dtype": str(self.dtype)
            },
            "training": {
                "learning_rate": self.lr,
                "effective_batch_size": self.get_effective_batch_size(),
                "max_iterations": self.max_iterations,
                "loss_type": self.loss_type
            },
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            },
            "lora": {
                "enabled": self.using_lora,
                "rank": self.lora_r if self.using_lora else None,
                "alpha": self.lora_alpha if self.using_lora else None
            }
        }
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        result = {}
        for field_name, field_obj in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if isinstance(value, torch.dtype):
                # Convert torch dtype to string for serialization
                dtype_map = {
                    torch.bfloat16: "bfloat16",
                    torch.float16: "float16", 
                    torch.float32: "float32",
                    torch.float64: "float64"
                }
                value = dtype_map.get(value, str(value))
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
    
    def validate_field_value(self, field_name: str, value: Any) -> bool:
        """Validate a value for a specific field."""
        if field_name not in self.__dataclass_fields__:
            return False
            
        field_obj = self.__dataclass_fields__[field_name]
        metadata = getattr(field_obj, 'metadata', {})
        
        # Check choices
        if 'choices' in metadata:
            return value in metadata['choices']
            
        # Check range
        if 'range' in metadata and isinstance(value, (int, float)):
            min_val, max_val = metadata['range']
            return min_val <= value <= max_val
            
        return True
    
    def print_config_summary(self):
        """Print a formatted summary of the configuration."""
        print("\\n" + "="*60)
        print("GRPO CONFIGURATION SUMMARY")
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
                if isinstance(value, torch.dtype):
                    value_str = str(value).split('.')[-1]
                elif isinstance(value, float):
                    value_str = f"{value:.6g}"
                else:
                    value_str = str(value)
                    
                print(f"  {field_name}: {value_str}")
                print(f"    {description}")
                
        print("\\n" + "="*60)