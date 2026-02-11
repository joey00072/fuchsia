import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import yaml

from fuchsia.rollout_queue import normalize_rollout_transfer_mode


@dataclass
class FuchsiaConfig:
    # Shared model configuration
    model: str = ""
    model_name: str = ""
    revision: Optional[str] = None
    model_revision: Optional[str] = None
    max_model_len: int = 1024
    dtype: str | torch.dtype = "bfloat16"

    # Server settings
    tensor_parallel_size: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.5
    enable_prefix_caching: Optional[bool] = None
    quantization: Optional[str] = None
    buffer_size: int = 32
    generation_batch_size: int = 1
    enable_lora: bool = False
    vllm_n: int = 1
    vllm_repetition_penalty: float = 1.0
    vllm_temperature: float = 0.9
    vllm_top_p: float = 1.0
    vllm_top_k: int = -1
    vllm_min_p: float = 0.0
    vllm_logprobs: int = 1
    vllm_max_tokens: int = 1024
    vllm_kv_quantization: bool = False

    # Dataset/transfer settings
    dataset_name: str = ""
    dataset_split: str = "train"
    dataset_max_samples: int = -1
    dataset_field: str = "prompt"
    sample_transfer_mode: str = "api"
    sample_transfer_dir: str = "/tmp/fuchsia_sample_queue"
    sample_transfer_poll_interval: float = 0.25
    sample_transfer_clear_on_start: bool = False

    # Shared rollout settings (consumed by trainer and server)
    group_size: int = 8

    # Trainer settings
    batch_size: int = 1
    # Deprecated alias for micro_batch_size; kept for backward compatibility with old YAMLs.
    grad_accumulation_steps: Optional[int] = None
    micro_batch_size: Optional[int] = None
    max_iterations: int = 1000000
    log_wandb: bool = False
    lr: float = 5e-6
    weight_decay: float = 0.0
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    wandb_project: str = "fuchsia"
    num_policy_updates: int = 8
    using_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: Optional[List[str]] = None
    lora_path: str = "lora_weights"
    ignore_imcomplete_samples: bool = False
    use_clipping: str = "ppo"
    use_scheduler: bool = True
    warmup_steps: int = 8
    scheduler_type: str = "constant_with_warmup"
    gradient_checkpointing_enabled: bool = False
    gradient_checkpointing_cpu_offloading: bool = False
    gradient_checkpointing_mode: str = "activation"
    max_new_tokens: int = 512
    temperature: float = 0.9
    repetition_penalty: float = 1.1
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    log_level: str = "INFO"
    save_every: int = 25
    async_buffer_fill: bool = True
    debug: bool = True
    debug_log_dir: str = "debug_logs"
    single_gpu: bool = False
    non_blocking: bool = False
    loss_type: str = "grpo"
    importance_sampling_correction: bool = True
    importance_ratio_type: str = "token"
    importance_token_mask_high: float = 8.0
    importance_token_mask_low: float = 0.125
    importance_sequence_clip_high: float = 10.0
    importance_geo_mask_high: float = 10.0
    importance_geo_mask_low: float = 0.1
    importance_sequence_mask_low: float = 0.0
    importance_sequence_mask_high: float = 100.0
    output_dir: str = "output"
    device: Optional[str] = None

    logger: logging.Logger = field(init=False, repr=False)
    _trainer_dtype: torch.dtype = field(init=False, repr=False)

    def __post_init__(self):
        # Keep naming aliases in sync.
        if not self.model_name and self.model:
            self.model_name = self.model
        if not self.model and self.model_name:
            self.model = self.model_name
        if self.model_revision is None and self.revision is not None:
            self.model_revision = self.revision
        if self.revision is None and self.model_revision is not None:
            self.revision = self.model_revision

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if isinstance(self.dtype, torch.dtype):
            trainer_dtype = self.dtype
            if trainer_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                trainer_dtype = torch.float16
            self._trainer_dtype = trainer_dtype
            inv = {v: k for k, v in dtype_map.items()}
            self.dtype = inv.get(self.dtype, "bfloat16")
        else:
            dtype_key = str(self.dtype).lower()
            if dtype_key == "auto":
                trainer_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                if dtype_key not in dtype_map:
                    raise ValueError(
                        f"Unsupported dtype: {self.dtype}. Supported values are: "
                        f"{list(dtype_map.keys()) + ['auto']}"
                    )
                trainer_dtype = dtype_map[dtype_key]
                if trainer_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    trainer_dtype = torch.float16
            self._trainer_dtype = trainer_dtype
            self.dtype = dtype_key

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
        if self.grad_accumulation_steps is not None and self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be > 0 when provided")
        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0 when provided")
        if self.micro_batch_size is not None and self.grad_accumulation_steps is not None:
            raise ValueError(
                "Ambiguous trainer batch config: set only `trainer.micro_batch_size` (preferred) "
                "or only legacy `trainer.grad_accumulation_steps`."
            )
        # Canonicalize checkpointing mode. Keep cpu_offloading as a legacy alias.
        mode = str(self.gradient_checkpointing_mode).lower()
        if mode == "activation" and self.gradient_checkpointing_cpu_offloading:
            mode = "activation_offload"
        mode_aliases = {
            "offload": "activation_offload",
            "cpu_offload": "activation_offload",
            "cpu_offloading": "activation_offload",
            "unsloth": "activation_offload",
        }
        mode = mode_aliases.get(mode, mode)
        allowed_modes = {"activation", "activation_offload"}
        if mode not in allowed_modes:
            raise ValueError(
                "Unsupported gradient_checkpointing.mode: "
                f"{self.gradient_checkpointing_mode}. "
                f"Supported values: {sorted(allowed_modes)}"
            )
        self.gradient_checkpointing_mode = mode
        self.gradient_checkpointing_cpu_offloading = (
            self.gradient_checkpointing_mode == "activation_offload"
        )
        if not self.debug_log_dir:
            self.debug_log_dir = "debug_logs"
        # Resolve relative debug log path under output_dir for predictable run artifacts.
        debug_path = Path(self.debug_log_dir)
        if not debug_path.is_absolute():
            self.debug_log_dir = str(Path(self.output_dir) / debug_path)

        logging.basicConfig(level=getattr(logging, self.log_level.upper(), logging.INFO))
        self.logger = logging.getLogger("Trainer")

    @property
    def trainer_dtype(self) -> torch.dtype:
        return self._trainer_dtype

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FuchsiaConfig":
        with open(yaml_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        default_max_iterations = cls.__dataclass_fields__["max_iterations"].default

        # Shared section (PrimeRL-style): values used by both trainer and server.
        # New preferred shape:
        # shared:
        #   rollout:
        #     group_size: 8
        #   sampling:
        #     max_tokens: 2048
        #     temperature: 0.8
        #     top_p: 1.0
        #     top_k: -1
        #     min_p: 0.0
        #     repetition_penalty: 1.0
        #     logprobs: 1
        #   runtime:
        #     single_gpu: true
        #     lora_path: ./lora_weights
        #   transfer:
        #     mode: api
        #     queue_dir: /tmp/fuchsia_sample_queue
        #     poll_interval: 0.25
        #     clear_on_start: false
        shared = config.get("shared", {})
        shared_rollout = shared.get("rollout", {})
        shared_sampling = shared.get("sampling", {})
        shared_runtime = shared.get("runtime", {})
        shared_transfer = shared.get("transfer", {})

        trainer = config.get("trainer", {})
        generation = config.get("generation", {})
        model = config.get("model", {})
        dataset = config.get("dataset", {})
        training = config.get("training", {})
        server = config.get("server", {})
        vllm = server.get("vllm", {})
        transfer = server.get("transfer", {})
        lora = config.get("lora", {})

        gradient_checkpointing = trainer.get("gradient_checkpointing", {})
        if isinstance(gradient_checkpointing, bool):
            gradient_checkpointing = {"enabled": gradient_checkpointing}
        importance = trainer.get("importance_sampling", {})

        def pick(*values, default=None):
            for value in values:
                if value is not None:
                    return value
            return default

        rollout_group_size = int(
            pick(
                shared_rollout.get("group_size"),
                shared_rollout.get("rollouts_per_example"),
                trainer.get("group_size"),
                generation.get("group_size"),
                vllm.get("n"),
                default=8,
            )
        )
        shared_max_tokens = pick(shared_sampling.get("max_tokens"), generation.get("max_len"))
        shared_temperature = pick(
            shared_sampling.get("temperature"),
            generation.get("temperature"),
        )
        shared_top_p = pick(shared_sampling.get("top_p"), generation.get("top_p"))
        shared_top_k = pick(shared_sampling.get("top_k"), generation.get("top_k"))
        shared_min_p = pick(shared_sampling.get("min_p"), generation.get("min_p"))
        shared_repetition_penalty = pick(
            shared_sampling.get("repetition_penalty"),
            generation.get("repetition_penalty"),
        )
        shared_logprobs = shared_sampling.get("logprobs")

        model_name = model.get("name", "")
        model_revision = model.get("revision")

        # Normalize legacy grad_accumulation_steps into micro_batch_size so trainer logic has one knob.
        grad_accumulation_steps = trainer.get("grad_accumulation_steps")
        micro_batch_size = trainer.get("micro_batch_size")
        if micro_batch_size is None and grad_accumulation_steps is not None:
            grad_steps = int(grad_accumulation_steps)
            micro_batch_size = max(1, (rollout_group_size + grad_steps - 1) // grad_steps)
            grad_accumulation_steps = None

        gc_cpu_offloading = bool(gradient_checkpointing.get("cpu_offloading", False))
        gc_mode_raw = gradient_checkpointing.get("mode")
        if gc_mode_raw is None:
            gc_mode = "activation_offload" if gc_cpu_offloading else "activation"
        else:
            gc_mode = str(gc_mode_raw).lower()

        return cls(
            model=model_name,
            model_name=model_name,
            revision=model_revision,
            model_revision=model_revision,
            dtype=model.get("dtype", "bfloat16"),
            max_model_len=model.get("max_model_len", 1024),

            host=server.get("host", "0.0.0.0"),
            port=server.get("port", 8000),
            gpu_memory_utilization=server.get("gpu_memory_utilization", 0.5),
            tensor_parallel_size=server.get("tensor_parallel_size", 1),
            enable_prefix_caching=server.get("enable_prefix_caching", None),
            quantization=server.get("quantization"),
            buffer_size=server.get("buffer_size", 32),
            generation_batch_size=server.get("generation_batch_size", 1),
            enable_lora=lora.get("enabled", False),
            vllm_n=int(pick(vllm.get("n"), rollout_group_size, default=1)),
            vllm_repetition_penalty=float(
                pick(vllm.get("repetition_penalty"), shared_repetition_penalty, default=1.0)
            ),
            vllm_temperature=float(
                pick(vllm.get("temperature"), shared_temperature, default=0.9)
            ),
            vllm_top_p=float(pick(vllm.get("top_p"), shared_top_p, default=1.0)),
            vllm_top_k=int(pick(vllm.get("top_k"), shared_top_k, default=-1)),
            vllm_min_p=float(pick(vllm.get("min_p"), shared_min_p, default=0.0)),
            vllm_logprobs=int(pick(vllm.get("logprobs"), shared_logprobs, default=1)),
            vllm_max_tokens=int(
                pick(vllm.get("max_tokens"), shared_max_tokens, default=1024)
            ),
            vllm_kv_quantization=vllm.get("kv_quantization", False),

            dataset_name=dataset.get("name", ""),
            dataset_split=dataset.get("split", "train"),
            dataset_max_samples=dataset.get("max_samples", -1),
            dataset_field=dataset.get("field", "prompt"),
            sample_transfer_mode=pick(
                shared_transfer.get("mode"),
                transfer.get("mode"),
                server.get("sample_transfer_mode"),
                default="api",
            ),
            sample_transfer_dir=pick(
                shared_transfer.get("queue_dir"),
                transfer.get("queue_dir"),
                server.get("sample_transfer_dir"),
                default="/tmp/fuchsia_sample_queue",
            ),
            sample_transfer_poll_interval=float(
                pick(
                    shared_transfer.get("poll_interval"),
                    transfer.get("poll_interval"),
                    server.get("sample_transfer_poll_interval"),
                    default=0.25,
                )
            ),
            sample_transfer_clear_on_start=bool(
                pick(
                    shared_transfer.get("clear_on_start"),
                    transfer.get("clear_on_start"),
                    server.get("sample_transfer_clear_on_start"),
                    default=False,
                )
            ),

            loss_type=trainer.get("loss_type", "grpo"),
            group_size=rollout_group_size,
            batch_size=trainer.get("batch_size", 1),
            grad_accumulation_steps=grad_accumulation_steps,
            micro_batch_size=micro_batch_size,
            lr=float(trainer.get("lr", 5e-6)),
            weight_decay=float(trainer.get("weight_decay", 0.0)),
            beta=float(trainer.get("beta", 0.0)),
            epsilon=float(trainer.get("epsilon", 0.2)),
            epsilon_high=float(trainer.get("epsilon_high", 0.28)),
            importance_sampling_correction=importance.get(
                "enabled", trainer.get("importance_sampling_correction", True)
            ),
            importance_ratio_type=importance.get(
                "ratio_type", trainer.get("importance_ratio_type", "token")
            ),
            importance_token_mask_high=float(
                importance.get(
                    "token_mask_high", trainer.get("importance_token_mask_high", 8.0)
                )
            ),
            importance_token_mask_low=float(
                importance.get(
                    "token_mask_low", trainer.get("importance_token_mask_low", 0.125)
                )
            ),
            importance_sequence_clip_high=float(
                importance.get(
                    "sequence_clip_high",
                    trainer.get("importance_sequence_clip_high", 10.0),
                )
            ),
            importance_geo_mask_high=float(
                importance.get(
                    "geo_mask_high", trainer.get("importance_geo_mask_high", 10.0)
                )
            ),
            importance_geo_mask_low=float(
                importance.get(
                    "geo_mask_low", trainer.get("importance_geo_mask_low", 0.1)
                )
            ),
            importance_sequence_mask_low=float(
                importance.get(
                    "sequence_mask_low", trainer.get("importance_sequence_mask_low", 0.0)
                )
            ),
            importance_sequence_mask_high=float(
                importance.get(
                    "sequence_mask_high", trainer.get("importance_sequence_mask_high", 100.0)
                )
            ),
            log_wandb=trainer.get("log_wandb", False),
            wandb_project=trainer.get("wandb_project", "fuchsia"),
            num_policy_updates=trainer.get("num_policy_updates", 8),
            lora_path=pick(shared_runtime.get("lora_path"), trainer.get("lora_path"), default="lora_weights"),
            single_gpu=bool(pick(shared_runtime.get("single_gpu"), trainer.get("single_gpu"), default=False)),
            use_scheduler=trainer.get("use_scheduler", True),
            warmup_steps=trainer.get("warmup_steps", 8),
            scheduler_type=trainer.get("scheduler_type", "constant_with_warmup"),
            gradient_checkpointing_enabled=gradient_checkpointing.get("enabled", False),
            gradient_checkpointing_cpu_offloading=gc_cpu_offloading,
            gradient_checkpointing_mode=gc_mode,
            max_new_tokens=int(pick(generation.get("max_len"), shared_max_tokens, vllm.get("max_tokens"), default=512)),
            temperature=float(pick(generation.get("temperature"), shared_temperature, vllm.get("temperature"), default=0.9)),
            repetition_penalty=float(
                pick(generation.get("repetition_penalty"), shared_repetition_penalty, default=1.1)
            ),
            top_p=float(pick(generation.get("top_p"), shared_top_p, vllm.get("top_p"), default=1.0)),
            top_k=int(pick(generation.get("top_k"), shared_top_k, vllm.get("top_k"), default=-1)),
            min_p=float(pick(generation.get("min_p"), shared_min_p, vllm.get("min_p"), default=0.0)),
            max_iterations=training.get("max_iterations", default_max_iterations),
            save_every=training.get("save_steps", 25),
            output_dir=training.get("output_dir", "output"),
            debug=bool(trainer.get("debug", True)),
            debug_log_dir=str(trainer.get("debug_log_dir", "debug_logs")),
            log_level=trainer.get("log_level", "INFO"),
            async_buffer_fill=bool(trainer.get("async_buffer_fill", True)),
            non_blocking=bool(trainer.get("non_blocking", False)),
            device=trainer.get("device"),
            using_lora=lora.get("enabled", False),
            lora_r=lora.get("r", 8),
            lora_alpha=lora.get("alpha", 16),
            lora_target_modules=lora.get("target_modules"),
        )
