import contextlib
import datetime
import gc
import logging
import os
import time
from numbers import Real
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict, Any, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from fuchsia.vllm_client import VLLMClient
from fuchsia.config import FuchsiaConfig

try:
    from torch.utils.viz._cycles import warn_tensor_cycles
    CYCLE_DETECTION_AVAILABLE = True
except ImportError:
    CYCLE_DETECTION_AVAILABLE = False

class Trainer:
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Optional[PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        dataset: Union[Iterator[Dict[str, Any]], IterableDataset],
        optimizer: Optional[torch.optim.Optimizer] = None,
        reward_functions: Optional[List[Callable]] = None,
        config: FuchsiaConfig = None,
        vllm_client: Optional[VLLMClient] = None,
    ) -> None:
        self.config = config
        self.logger = getattr(config, "logger", logging.getLogger("Trainer"))
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.debug_enabled = bool(getattr(config, "debug", False))
        self.debug_logger: logging.Logger | None = None
        self.debug_log_path: str | None = None
        self._setup_debug_logger()
        
        self.vllm_client = vllm_client if vllm_client is not None else VLLMClient()

        # Enable reference cycle detection if available
        if CYCLE_DETECTION_AVAILABLE:
            warn_tensor_cycles()
            self.logger.info("Reference cycle detection enabled")
        
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = model.name_or_path

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="cpu",
            torch_dtype=config.dtype
        ) if isinstance(model, str) else model
        
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        if isinstance(dataset, IterableDataset):
            # Dataset emits already-prepared trainer batches.
            self.data_loader = DataLoader(dataset, batch_size=None)
            self.data_loader_iter = iter(self.data_loader)
        else:
            self.data_loader = None
            self.data_loader_iter = iter(self.dataset)
        if config.group_size <= 0:
            raise ValueError("group_size must be > 0")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.group_size = int(config.group_size)
        self.batch_size = int(config.batch_size)
        self.micro_batch_size_override = (
            int(config.micro_batch_size) if getattr(config, "micro_batch_size", None) is not None else None
        )
        self.max_iterations = config.max_iterations
        self.dtype = config.trainer_dtype
        self.beta = config.beta
        self.epsilon = config.epsilon
        self.epsilon_high = config.epsilon_high
        self.single_gpu = config.single_gpu
        self.non_blocking = False # config.non_blocking
        print("LOSS TYPE: ", config.loss_type)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                self.model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
        )
        
        # Initialize learning rate scheduler
        self.use_scheduler = config.use_scheduler
        self.scheduler = None
        if self.use_scheduler:
            self._setup_scheduler(config)
        
        self.logger.info(f"Learning rate: {config.lr}")
        if self.use_scheduler:
            self.logger.info(f"Scheduler: {config.scheduler_type} with {config.warmup_steps} warmup steps")
        self.reward_functions: list = reward_functions
        self.dataset_field = config.dataset_field
        self.num_policy_updates = config.num_policy_updates

        self.using_lora = config.using_lora
        self.lora_path = config.lora_path
        self.ignore_imcomplete_samples = config.ignore_imcomplete_samples
        self.async_buffer_fill = config.async_buffer_fill
        
        # Memory management state
        self._is_model_on_gpu = False
        self._is_optimizer_on_gpu = False
        
        if self.using_lora and self.beta > 0:
            self.ref_model = model
        
        self.model.to(self.dtype)
        if self.beta > 0 and ref_model is not None:
            self.ref_model = self.model
        
        # Set up proper memory management
        self._setup_memory_management()

        self.log_wandb = config.log_wandb
        if self.log_wandb:
            wandb.init(project=config.wandb_project)

        self.metrics: dict[str, float] = {}
        self._last_logged_step = 0

        self._debug_log(
            "trainer_init model=%s device=%s batch_size=%s group_size=%s micro_batch_size=%s "
            "single_gpu=%s output_dir=%s",
            self.model_name,
            self.device,
            self.batch_size,
            self.group_size,
            self.micro_batch_size_override,
            self.single_gpu,
            getattr(self.config, "output_dir", "output"),
        )
        self._debug_gpu_memory("trainer_init")

    def _setup_debug_logger(self) -> None:
        if not self.debug_enabled:
            return
        try:
            base_dir = Path(getattr(self.config, "debug_log_dir", "debug_logs"))
            base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique = f"{timestamp}_{os.getpid()}_{id(self) & 0xFFFF:04x}"
            log_path = base_dir / f"trainer_debug_{unique}.log"

            logger_name = f"TrainerDebug.{os.getpid()}.{id(self)}"
            self.debug_logger = logging.getLogger(logger_name)
            self.debug_logger.setLevel(logging.DEBUG)
            self.debug_logger.propagate = False
            self.debug_logger.handlers.clear()

            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.debug_logger.addHandler(handler)
            self.debug_log_path = str(log_path)
            self.logger.info("Debug diagnostics log: %s", self.debug_log_path)
        except Exception as exc:
            self.logger.warning("Failed to initialize debug logger: %s", exc)
            self.debug_logger = None
            self.debug_log_path = None

    def _debug_log(self, message: str, *args: Any) -> None:
        if self.debug_logger is not None:
            self.debug_logger.debug(message, *args)

    def _debug_gpu_memory(self, tag: str, **fields: Any) -> None:
        if not self.debug_enabled:
            return
        if not torch.cuda.is_available():
            self._debug_log("gpu_memory tag=%s cuda_available=false", tag)
            return
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
        max_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        payload = " ".join(f"{k}={v}" for k, v in fields.items())
        self._debug_log(
            "gpu_memory tag=%s alloc_gb=%.3f reserved_gb=%.3f max_alloc_gb=%.3f "
            "max_reserved_gb=%.3f free_gb=%.3f total_gb=%.3f %s",
            tag,
            alloc_gb,
            reserved_gb,
            max_alloc_gb,
            max_reserved_gb,
            free_bytes / (1024**3),
            total_bytes / (1024**3),
            payload,
        )

    @staticmethod
    def _is_numeric_scalar(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _current_learning_rate(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _compute_grad_norm(self) -> float:
        total_sq = 0.0
        for param in self.model.parameters():
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            grad_norm = grad.detach().float().norm(2).item()
            total_sq += grad_norm * grad_norm
        return total_sq ** 0.5

    def _collect_gpu_metrics(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {
                "gpu/allocated_gb": 0.0,
                "gpu/reserved_gb": 0.0,
                "gpu/free_gb": 0.0,
                "gpu/total_gb": 0.0,
                "gpu/peak_allocated_gb": 0.0,
                "gpu/peak_reserved_gb": 0.0,
            }
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "gpu/allocated_gb": float(torch.cuda.memory_allocated() / (1024**3)),
            "gpu/reserved_gb": float(torch.cuda.memory_reserved() / (1024**3)),
            "gpu/free_gb": float(free_bytes / (1024**3)),
            "gpu/total_gb": float(total_bytes / (1024**3)),
            "gpu/peak_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
            "gpu/peak_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024**3)),
        }

    def _log_to_wandb(self, metrics: dict[str, Any], *, step: Optional[int] = None) -> None:
        if not self.log_wandb:
            return
        payload = {k: float(v) for k, v in metrics.items() if self._is_numeric_scalar(v)}
        if not payload:
            return
        if step is None:
            wandb.log(payload)
            return
        wandb.log(payload, step=step)

    def _record_step_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        self._last_logged_step = step
        self.metrics["idx"] = float(step)
        for key, value in metrics.items():
            if self._is_numeric_scalar(value):
                self.metrics[key] = float(value)
        self._log_to_wandb(metrics, step=step)

    def _setup_memory_management(self) -> None:
        """Setup CUDA memory management for optimal performance."""
        if not torch.cuda.is_available():
            return
            
        # Configure CUDA allocator for better memory management
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable expandable segments if available
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            self.logger.info("Enabled CUDA expandable segments")
        except Exception as e:
            self.logger.debug(f"Could not enable expandable segments: {e}")
            
        # Log initial memory state
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        self.logger.info(f"Initial GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def _setup_scheduler(self, config: FuchsiaConfig) -> None:
        """Setup learning rate scheduler based on configuration."""
        if config.scheduler_type == "constant_with_warmup":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps
            )
        elif config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_iterations
            )
        elif config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.max_iterations
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")
        
        self.logger.info(f"Initialized {config.scheduler_type} scheduler with {config.warmup_steps} warmup steps")

    @torch.no_grad()
    def offload_to_cpu(self) -> PreTrainedModel:
        """Improved CPU offloading with proper memory management."""
        offload_start_time = time.perf_counter()
        self.logger.info("Starting offload to CPU...")
        
        # Wake CUDA allocator (safety net)
        torch.randn(1).cuda()

        for param in self.model.parameters():
            param.data = param.data.to("cpu", non_blocking=self.non_blocking)
            if hasattr(param, "_local_shard"): # this need for fsdp
                param._local_shard = param.data
            if param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=self.non_blocking)
                
        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cpu", non_blocking=self.non_blocking)
            
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu", non_blocking=self.non_blocking)
        
        self.model.eval()        
        self._is_model_on_gpu = False
        self._is_optimizer_on_gpu = False
        
        self.clean_and_sync_memory()
        
        offload_time = time.perf_counter() - offload_start_time
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        self.logger.info(f"CPU offload completed in {offload_time:.2f}s - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        return self.model
    
    @torch.no_grad()
    def load_model_to_gpu(self) -> PreTrainedModel:
        gpu_load_start_time = time.perf_counter()
        self.logger.info("Starting model load to GPU...")
        
        # Wake CUDA allocator (safety net)
        torch.randn(1).cuda()

        for param in self.model.parameters():
            param.data = param.data.to("cuda", non_blocking=self.non_blocking)
            
            if hasattr(param, "_local_shard"): # this need for fsdp
                param._local_shard = param.data
            if param.grad is not None:
                param.grad = param.grad.to("cuda", non_blocking=self.non_blocking)
                
        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cuda", non_blocking=self.non_blocking)
            
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to("cuda", non_blocking=self.non_blocking)
    
        self._is_model_on_gpu = True
        self._is_optimizer_on_gpu = True
        
        self.clean_and_sync_memory()
        
        gpu_load_time = time.perf_counter() - gpu_load_start_time
        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_memory_gb = torch.cuda.memory_reserved() / (1024**3)
        self.logger.info(f"GPU load completed in {gpu_load_time:.2f}s - GPU Memory: {allocated_memory_gb:.2f}GB allocated, {reserved_memory_gb:.2f}GB reserved")
        
        return self.model

    @torch.no_grad()
    def move_optimizer_to_cpu(self) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to("cpu", non_blocking=self.non_blocking)
        
        self._is_optimizer_on_gpu = False
        self.clean_and_sync_memory()

    @torch.no_grad()
    def cleanup_tensors(self, *tensors: Tensor) -> None:
        for tensor in tensors:
            if hasattr(tensor, 'data'):
                del tensor
        self.clean_and_sync_memory()
        
    def clean_and_sync_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        self.logger.info("Cleaning and syncing memory...")
        torch.cuda.synchronize()
        # Wake CUDA allocator (safety net)
        torch.randn(1).cuda()
        gc.collect()
        torch.cuda.empty_cache()

    def _wait_for_trainer_memory_release(
        self,
        baseline_allocated_bytes: int,
        baseline_reserved_bytes: int,
        timeout: float = 30.0,
        poll_interval: float = 0.25,
    ) -> bool:
        """
        Wait until trainer-process CUDA memory drops significantly after offload.
        """
        if not torch.cuda.is_available():
            return True

        # Require strong drop from baseline while allowing small allocator residue.
        target_allocated_bytes = min(int(baseline_allocated_bytes * 0.15), 512 * 1024 * 1024)  # <=15% or <=512MB
        target_reserved_bytes = min(int(baseline_reserved_bytes * 0.35), 2 * 1024 * 1024 * 1024)  # <=35% or <=2GB
        target_allocated_bytes = max(target_allocated_bytes, 64 * 1024 * 1024)  # floor 64MB
        target_reserved_bytes = max(target_reserved_bytes, 256 * 1024 * 1024)   # floor 256MB

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if allocated <= target_allocated_bytes and reserved <= target_reserved_bytes:
                self.logger.info(
                    "Trainer CUDA memory released: allocated %.2fGB, reserved %.2fGB",
                    allocated / (1024**3),
                    reserved / (1024**3),
                )
                return True
            self.logger.info(
                "Waiting trainer memory release: allocated %.2fGB (<= %.2fGB), reserved %.2fGB (<= %.2fGB)",
                allocated / (1024**3),
                target_allocated_bytes / (1024**3),
                reserved / (1024**3),
                target_reserved_bytes / (1024**3),
            )
            time.sleep(poll_interval)
        return False

    def _wait_for_global_free_memory(
        self,
        min_free_bytes: int,
        timeout: float = 120.0,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait for global device free memory (across processes) to reach a threshold."""
        if not torch.cuda.is_available():
            return True

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= min_free_bytes:
                self.logger.info(
                    "Global free GPU memory recovered: %.2fGB / %.2fGB",
                    free_bytes / (1024**3),
                    total_bytes / (1024**3),
                )
                return True
            self.logger.info(
                "Waiting for global GPU free memory: %.2fGB / %.2fGB (target %.2fGB)",
                free_bytes / (1024**3),
                total_bytes / (1024**3),
                min_free_bytes / (1024**3),
            )
            time.sleep(poll_interval)
        return False

    def selective_log_softmax(
        self,
        logits: Tensor,                 # (B, T, V)
        labels: Tensor,                 # (B, T)
        *,
        chunk_size: int = 32,           # 0 ⇒ disable chunking
        ignore_index: int = -100,
    ) -> Tensor:
        """
        Returns per‑token log‑probs log p(x_t = labels_t | x_<t>)
        while keeping peak memory low by processing `chunk_size`
        rows at a time (rows = B·T tokens).

        The fast‑path (chunk_size==0) is fully vectorised.
        """

        B, T, V = logits.shape
        # Keep the original 3D layout. Flatten+reshape can materialize a full copy when
        # logits are non-contiguous (common with fused attention kernels), causing large spikes.
        if chunk_size == 0:
            safe_labels = labels.clamp_min(0).unsqueeze(-1)
            picked = logits.gather(-1, safe_labels).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            return torch.where(labels == ignore_index, torch.zeros_like(picked), picked)

        out: list[Tensor] = []
        rows_per_step = max(int(chunk_size), 1)
        tokens_per_step = max(1, rows_per_step // max(B, 1))
        for token_start in range(0, T, tokens_per_step):
            token_end = min(token_start + tokens_per_step, T)
            logits_chunk = logits[:, token_start:token_end, :]
            labels_chunk = labels[:, token_start:token_end]
            safe_labels = labels_chunk.clamp_min(0).unsqueeze(-1)
            picked_chunk = logits_chunk.gather(-1, safe_labels).squeeze(-1) - torch.logsumexp(
                logits_chunk, dim=-1
            )
            picked_chunk = torch.where(
                labels_chunk == ignore_index,
                torch.zeros_like(picked_chunk),
                picked_chunk,
            )
            out.append(picked_chunk)

        return torch.cat(out, dim=1)

    def get_per_token_logps(
        self,
        model,
        input_ids: Tensor,
        *,
        chunk_size: int = 32,
        training: bool = False,
    ) -> Tensor:
        """
        Computes log‑probs for the *next* token at every position
        (causal language‑model shift).
        """
        logits  = model(input_ids=input_ids, training=training).logits   # (B, T, V)
        logits  = logits[:, :-1, :]
        labels  = input_ids[:, 1:]
        return self.selective_log_softmax(logits, labels,
                                    chunk_size=chunk_size,
                                    ignore_index=-100)

    def _apply_importance_sampling_correction(
        self,
        policy_log_probs: Tensor,
        old_policy_log_probs: Tensor,
        loss_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        log_importance_ratio = policy_log_probs - old_policy_log_probs.detach()
        token_importance_ratio = torch.exp(log_importance_ratio)

        if not self.config.importance_sampling_correction:
            return token_importance_ratio, loss_mask

        loss_mask = loss_mask.bool()

        seq_denom = loss_mask.sum(dim=-1).clamp_min(1).to(log_importance_ratio.dtype)
        masked_log_ratio_sum = (log_importance_ratio * loss_mask.to(log_importance_ratio.dtype)).sum(dim=-1)
        geo_seq_ratio = torch.exp(masked_log_ratio_sum / seq_denom)

        seq_log_importance_ratio = torch.clamp(masked_log_ratio_sum.detach(), max=10.0)
        seq_importance_ratio = torch.clamp(
            torch.exp(seq_log_importance_ratio),
            max=self.config.importance_sequence_clip_high,
        )

        seq_min_ratio = torch.where(loss_mask, token_importance_ratio, torch.inf).amin(dim=-1)
        seq_max_ratio = torch.where(loss_mask, token_importance_ratio, -torch.inf).amax(dim=-1)

        token_mask_low = token_importance_ratio < self.config.importance_token_mask_low
        token_mask_high = token_importance_ratio > self.config.importance_token_mask_high
        geo_mask_low = geo_seq_ratio < self.config.importance_geo_mask_low
        geo_mask_high = geo_seq_ratio > self.config.importance_geo_mask_high
        seq_mask_low = seq_min_ratio < self.config.importance_sequence_mask_low
        seq_mask_high = seq_max_ratio > self.config.importance_sequence_mask_high

        is_masked = token_mask_low | token_mask_high
        is_masked = (
            is_masked
            | geo_mask_low.unsqueeze(-1)
            | geo_mask_high.unsqueeze(-1)
            | seq_mask_low.unsqueeze(-1)
            | seq_mask_high.unsqueeze(-1)
        )
        keep_mask = loss_mask & ~is_masked

        if self.config.importance_ratio_type == "sequence":
            importance_ratio = seq_importance_ratio.unsqueeze(-1).expand_as(token_importance_ratio)
        else:
            importance_ratio = token_importance_ratio

        return importance_ratio, keep_mask
    
    def compute_loss(
        self, 
        inputs: Tensor, 
        old_policy_log_probs: Tensor, 
        reward: Tensor, 
        mean_rewards: Tensor, 
        std_rewards: Tensor, 
        loss_mask: Tensor, 
        ignore_sample: Tensor
    ) -> Tuple[Tensor, float]:
        
        if self.beta != 0:
            with torch.no_grad():
                # calculating this before calculating policy_log_probs reduces peak memory usage
                with (
                    self.ref_model.disable_adapter()
                    if self.using_lora
                    else contextlib.nullcontext()
                ):
                    ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)
            
        policy_log_probs = self.get_per_token_logps(self.model, inputs, training=True)

        kld = 0
        
        if self.beta != 0:
            # kl divergence calculation
            log_ratios = ref_policy_log_probs - policy_log_probs
            kld = torch.exp(log_ratios) - log_ratios - 1

        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)
        # advantage = torch.where(advantage < 0, advantage * 0.001, advantage)
        
        policy_ratio, effective_loss_mask = self._apply_importance_sampling_correction(
            policy_log_probs, old_policy_log_probs, loss_mask
        )
        

        unclipped_loss = policy_ratio * advantage
        clipped_loss = (
            torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon_high) * advantage
        )
        loss = -torch.min(unclipped_loss, clipped_loss)
        
        if self.config.loss_type == "cxpo":
            loss = F.sigmoid((policy_ratio))*advantage
            return loss.mean(), 0.0
        
        if self.config.loss_type == "cispo":
            loss = -clipped_loss.detach() * policy_log_probs
            loss = (loss * effective_loss_mask).sum(dim=-1) / (effective_loss_mask.sum(dim=-1) + 1e-6)
            return loss.mean(), 0.0  # Return 0.0 for KLD when using CISPO
        elif self.config.loss_type == "reinforce":
            loss = - unclipped_loss
        
        if self.ignore_imcomplete_samples:
            loss = loss * ignore_sample
            kld = kld * ignore_sample
        
        loss = (loss * effective_loss_mask).sum(dim=-1) / (effective_loss_mask.sum(dim=-1) + 1e-6)
        kld =  (kld  * effective_loss_mask).sum(dim=-1) / (effective_loss_mask.sum(dim=-1) + 1e-6)

        loss += kld * self.beta

        # Calculate average KLD for this batch
        avg_kld = kld.mean().item() if self.beta != 0 else 0.0

        return loss.mean(), avg_kld

    def sample_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict, dict[str, float]]:
        prepared = next(self.data_loader_iter)
        if not isinstance(prepared, dict):
            raise RuntimeError(
                f"Prepared dataset must yield dict batches, got {type(prepared)}"
            )

        batch_metrics: dict[str, float] = {}
        metrics = prepared.get("metrics", {})
        if isinstance(metrics, dict):
            prompts = metrics.get("prompt")
            completions = metrics.get("completions")
            if prompts is not None and completions is not None:
                batch_metrics["rollout/prompt_count"] = float(len(prompts))
                batch_metrics["rollout/completion_count"] = float(len(completions))
            avg_token_lengths = metrics.get("avg_token_lengths")
            if avg_token_lengths is not None:
                batch_metrics["rollout/avg_completion_tokens"] = float(avg_token_lengths)

            if self.config.debug:
                decoded_outputs = metrics.get("decoded_outputs") or []
                if decoded_outputs:
                    print("\n\n\n")
                    print("-" * 10)
                    print(
                        decoded_outputs[0]
                        .replace("<|endoftext|>", "")
                        .replace("<|finetune_right_pad_id|>", "")
                        .replace("<|end_of_text|>", "")
                    )
                    print("-" * 10)
                    print("\n\n\n")

        server_info = prepared.get("server_info", {})
        if not isinstance(server_info, dict):
            server_info = {}

        return (
            prepared["outputs"],
            prepared["rewards"],
            prepared["mean_rewards"],
            prepared["std_rewards"],
            prepared["loss_mask"],
            prepared["ignore_sample"],
            server_info,
            batch_metrics,
        )

    def log_metrics(self) -> None:
        if not self.log_wandb:
            return
            
        current_idx = int(self.metrics.get("idx", 0.0))
        
        metrics = {
            k: float(v)
            for k, v in self.metrics.items()
            if k != "idx" and self._is_numeric_scalar(v)
        }
        metrics["train/iteration"] = float(current_idx)
        
        self._log_to_wandb(metrics, step=current_idx)

    def handle_policy_update(self) -> None:
        self.vllm_client.update_model_params(
            self.model, self.tokenizer, lora=self.using_lora, 
            single_gpu=self.single_gpu, lora_path=self.lora_path
        )
        
        if not self.async_buffer_fill:
            self.vllm_client.empty_buffer()
            
        if not self.single_gpu:
            self.vllm_client.fill_buffer()
            return
            
        # if hotswap is enabled
        self._perform_vllm_hotswap_cycle()

    def _perform_vllm_hotswap_cycle(self) -> None:
        """Perform VLLM hotswap cycle without fixed sleeps by polling explicit server/global memory state."""
        vllm_cycle_start_time = time.perf_counter()
        self.logger.info("Starting VLLM hotswap cycle...")
        self._debug_gpu_memory("hotswap_start")
        pre_status = self.vllm_client.buffer_status()
        self._debug_log("hotswap_status phase=start status=%s", pre_status)
        pre_buffer_size = float(pre_status.get("current_size", 0.0)) if isinstance(pre_status, dict) else 0.0

        trainer_alloc_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        trainer_reserved_before = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        pre_offload_free_bytes = None
        total_bytes = None
        if torch.cuda.is_available():
            pre_offload_free_bytes, total_bytes = torch.cuda.mem_get_info()
            self.logger.info(
                "Global free GPU memory before trainer offload: %.2fGB / %.2fGB",
                pre_offload_free_bytes / (1024**3),
                total_bytes / (1024**3),
            )

        self.offload_to_cpu()
        self._debug_gpu_memory("after_offload_to_cpu")
        if torch.cuda.is_available():
            if not self._wait_for_trainer_memory_release(
                baseline_allocated_bytes=trainer_alloc_before,
                baseline_reserved_bytes=trainer_reserved_before,
                timeout=45.0,
                poll_interval=0.25,
            ):
                raise RuntimeError(
                    "Trainer model memory did not release sufficiently after offload; aborting hotswap reload cycle"
                )

        baseline_free_bytes = None
        if torch.cuda.is_available():
            if pre_offload_free_bytes is not None and total_bytes is not None:
                expected_recovery_bytes = max(
                    int(trainer_alloc_before * 0.70),
                    256 * 1024 * 1024,
                )
                min_free_before_wake = min(
                    pre_offload_free_bytes + expected_recovery_bytes,
                    int(total_bytes * 0.97),
                )
                min_free_before_wake = max(min_free_before_wake, pre_offload_free_bytes)
                if not self._wait_for_global_free_memory(
                    min_free_bytes=min_free_before_wake,
                    timeout=90.0,
                    poll_interval=0.25,
                ):
                    raise RuntimeError(
                        "Global free GPU memory did not recover enough after trainer offload; "
                        "aborting wake-up to avoid overlapping allocations"
                    )
            baseline_free_bytes, total_bytes = torch.cuda.mem_get_info()
            self.logger.info(
                "Baseline global free GPU memory after trainer offload: %.2fGB / %.2fGB",
                baseline_free_bytes / (1024**3),
                total_bytes / (1024**3),
            )

        vllm_wake_start_time = time.perf_counter()
        wake_response = self.vllm_client.wake_up()
        if isinstance(wake_response, dict) and wake_response.get("error"):
            raise RuntimeError(f"Failed to wake up VLLM server: {wake_response['error']}")
        if not self.vllm_client.wait_until_awake(timeout=120.0, poll_interval=1.0):
            raise RuntimeError("VLLM server did not reach steady awake state before buffer fill")
        self._debug_log("hotswap_status phase=awake status=%s", self.vllm_client.buffer_status())
        self._debug_gpu_memory("after_wake")

        # Ensure the queue has enough samples for at least one trainer batch before sleeping again.
        min_buffer_items = max(1, min(self.batch_size, getattr(self.config, "buffer_size", self.batch_size)))
        buffer_ready = False
        fill_attempts = 3
        for attempt in range(1, fill_attempts + 1):
            pre_status = self.vllm_client.buffer_status()
            already_filling = False
            if isinstance(pre_status, dict):
                already_filling = bool(
                    pre_status.get("is_filling", False)
                    or pre_status.get("fill_slot_claimed", False)
                )
            assume_filling = already_filling
            if not already_filling:
                fill_response = self.vllm_client.fill_buffer()
                if isinstance(fill_response, dict) and fill_response.get("error"):
                    self.logger.warning(
                        "VLLM buffer fill request returned error on attempt %s/%s: %s",
                        attempt,
                        fill_attempts,
                        fill_response["error"],
                    )
                elif isinstance(fill_response, dict):
                    response_message = str(fill_response.get("message", "")).lower()
                    if "start" in response_message or "progress" in response_message:
                        assume_filling = True
            else:
                self.logger.info(
                    "Buffer fill already in progress before attempt %s/%s (status=%s)",
                    attempt,
                    fill_attempts,
                    pre_status,
                )

            wait_timeout = 180.0 if attempt == 1 else 120.0
            if self.vllm_client.wait_for_buffer_ready(
                min_size=min_buffer_items,
                timeout=wait_timeout,
                poll_interval=1.0,
                filling_grace_timeout=900.0,
                assume_filling=assume_filling,
            ):
                buffer_ready = True
                break
            status = self.vllm_client.buffer_status()
            self.logger.warning(
                "Buffer not ready after fill attempt %s/%s (status=%s)",
                attempt,
                fill_attempts,
                status,
            )

        if not buffer_ready:
            raise RuntimeError(
                f"Buffer failed to reach ready state (min_size={min_buffer_items}) before sleep; "
                "aborting hotswap cycle to avoid trainer stall"
            )
        buffer_ready_status = self.vllm_client.buffer_status()
        self._debug_log("hotswap_status phase=buffer_ready status=%s", buffer_ready_status)
        ready_buffer_size = (
            float(buffer_ready_status.get("current_size", 0.0))
            if isinstance(buffer_ready_status, dict)
            else 0.0
        )

        sleep_response = self.vllm_client.sleep(max_retries=40, retry_sleep_time=1, max_retry_sleep_time=4)
        if not (isinstance(sleep_response, dict) and sleep_response.get("sleep", False)):
            raise RuntimeError(f"Failed to put VLLM server to sleep: {sleep_response}")

        if not self.vllm_client.wait_until_sleeping(timeout=120.0, poll_interval=1.0):
            raise RuntimeError("VLLM server did not reach steady sleeping state before trainer reload")
        self._debug_log("hotswap_status phase=sleeping status=%s", self.vllm_client.buffer_status())
        self._debug_gpu_memory("after_sleep")

        if baseline_free_bytes is not None:
            # vLLM should release most of the memory it used while awake.
            target_free_bytes = int(baseline_free_bytes * 0.95)
            if not self._wait_for_global_free_memory(
                min_free_bytes=target_free_bytes,
                timeout=120.0,
                poll_interval=0.5,
            ):
                raise RuntimeError(
                    "Global free GPU memory did not recover to target "
                    f"{target_free_bytes / (1024**3):.2f}GB before trainer reload"
                )
            
        vllm_wake_time = time.perf_counter() - vllm_wake_start_time
        self.load_model_to_gpu()
        self._debug_gpu_memory("after_model_reload")
        total_vllm_cycle_time = time.perf_counter() - vllm_cycle_start_time
        self.logger.info(f"VLLM hotswap cycle completed in {total_vllm_cycle_time:.2f}s (VLLM wake/fill/sleep: {vllm_wake_time:.2f}s)")
        self._debug_log(
            "hotswap_timing total_seconds=%.3f wake_fill_sleep_seconds=%.3f",
            total_vllm_cycle_time,
            vllm_wake_time,
        )
        self._log_to_wandb(
            {
                "sync/hotswap_total_seconds": total_vllm_cycle_time,
                "sync/vllm_wake_fill_sleep_seconds": vllm_wake_time,
                "sync/buffer_size_before_wake": pre_buffer_size,
                "sync/buffer_size_before_sleep": ready_buffer_size,
                "sync/min_buffer_items_target": float(min_buffer_items),
            },
            step=self._last_logged_step if self._last_logged_step > 0 else None,
        )

    def _resolve_group_micro_batch_size(self, group_size: int) -> int:
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")
        if self.micro_batch_size_override is not None:
            resolved = min(group_size, self.micro_batch_size_override)
            self._debug_log(
                "microbatch_resolve source=config group_size=%s configured=%s resolved=%s",
                group_size,
                self.micro_batch_size_override,
                resolved,
            )
            return resolved
        self._debug_log(
            "microbatch_resolve source=default group_size=%s resolved=%s",
            group_size,
            group_size,
        )
        # Default: process full rollout group as one microbatch when override is not set.
        return group_size

    @staticmethod
    def _iter_micro_ranges(num_rows: int, micro_batch_size: int):
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be > 0, got {micro_batch_size}")
        for start in range(0, num_rows, micro_batch_size):
            yield start, min(start + micro_batch_size, num_rows)

    def _compute_old_policy_log_probs(
        self,
        batch_inputs: Tensor,
        micro_batch_size: int,
    ) -> list[Tensor]:
        pi_old: list[Tensor] = []
        with torch.no_grad():
            for group_inputs in batch_inputs:
                group_log_probs: list[Tensor] = []
                for start, end in self._iter_micro_ranges(group_inputs.shape[0], micro_batch_size):
                    micro_inputs = group_inputs[start:end]
                    micro_log_probs = self.get_per_token_logps(self.model, micro_inputs)
                    group_log_probs.append(micro_log_probs)
                pi_old.append(torch.cat(group_log_probs, dim=0))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pi_old

    def _step_scheduler(self) -> float:
        if self.use_scheduler and self.scheduler is not None:
            self.scheduler.step()
            current_lr = float(self.scheduler.get_last_lr()[0])
        else:
            current_lr = self._current_learning_rate()
        self.logger.debug(f"Current learning rate: {current_lr:.2e}")
        return current_lr

    def _train_single_group(
        self,
        b_inputs: Tensor,
        b_old_policy_log_probs: Tensor,
        b_reward: Tensor,
        b_loss_mask: Tensor,
        b_ignore_sample: Tensor,
        b_mean_rewards: Tensor,
        b_std_rewards: Tensor,
        micro_batch_size: int,
    ) -> dict[str, float]:
        micro_ranges = list(self._iter_micro_ranges(b_inputs.shape[0], micro_batch_size))
        if not micro_ranges:
            raise RuntimeError("No microbatches produced for group training step")
        step_started_at = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        microbatch_share = micro_batch_size / max(int(b_inputs.shape[0]), 1)
        self._debug_log(
            "group_step rows=%s micro_batch_size=%s micro_steps=%s microbatch_share=%.4f seq_len=%s",
            b_inputs.shape[0],
            micro_batch_size,
            len(micro_ranges),
            microbatch_share,
            b_inputs.shape[1] if b_inputs.ndim >= 2 else -1,
        )
        self._debug_gpu_memory(
            "group_step_start",
            rows=b_inputs.shape[0],
            micro_batch_size=micro_batch_size,
            micro_steps=len(micro_ranges),
        )

        self.optimizer.zero_grad(set_to_none=True)

        group_losses: list[float] = []
        group_klds: list[float] = []
        num_micro_steps = len(micro_ranges)
        tokens_in_step = int(b_loss_mask.sum().item()) if torch.is_tensor(b_loss_mask) else 0
        samples_in_step = int(b_inputs.shape[0])

        for start, end in micro_ranges:
            inputs = b_inputs[start:end]
            old_policy_log_probs = b_old_policy_log_probs[start:end]
            reward_batch = b_reward[start:end]
            loss_mask_batch = b_loss_mask[start:end]
            ignore_sample_batch = b_ignore_sample[start:end].unsqueeze(-1)
            mean_rewards = b_mean_rewards[start:end]
            std_rewards = b_std_rewards[start:end]

            loss, kld = self.compute_loss(
                inputs,
                old_policy_log_probs,
                reward_batch,
                mean_rewards,
                std_rewards,
                loss_mask_batch,
                ignore_sample_batch,
            )

            group_losses.append(loss.item())
            group_klds.append(kld)
            (loss / num_micro_steps).backward()

        grad_norm = self._compute_grad_norm()
        self.optimizer.step()
        current_lr = self._step_scheduler()
        step_seconds = max(time.perf_counter() - step_started_at, 1e-9)
        throughput_tokens = tokens_in_step / step_seconds
        throughput_samples = samples_in_step / step_seconds
        gpu_metrics = self._collect_gpu_metrics()
        self._debug_gpu_memory("group_step_end")

        avg_loss = sum(group_losses) / len(group_losses)
        avg_kld = sum(group_klds) / len(group_klds)
        step_metrics = {
            "loss": avg_loss,
            "kld": avg_kld,
            "total_reward": b_reward.mean().item(),
            "mean_group_reward": b_mean_rewards.mean().item(),
            "valid_samples": b_ignore_sample.sum().item(),
            "optim/learning_rate": current_lr,
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm,
            "perf/step_seconds": step_seconds,
            "perf/tokens_per_step": float(tokens_in_step),
            "perf/samples_per_step": float(samples_in_step),
            "perf/throughput_tokens_per_sec": throughput_tokens,
            "perf/throughput": throughput_tokens,
            "perf/throughput_samples_per_sec": throughput_samples,
            "perf/micro_steps": float(num_micro_steps),
            "perf/micro_batch_size": float(micro_batch_size),
            "perf/microbatch_share": microbatch_share,
        }
        step_metrics.update(gpu_metrics)
        step_metrics["perf/peak_memory_gb"] = step_metrics.get("gpu/peak_reserved_gb", 0.0)
        return step_metrics

    def train(self, epochs: int = 1, max_iterations: Optional[int] = None) -> None:
        idx = 0
        start_time = time.perf_counter()
        target_max_iterations = self.max_iterations if max_iterations is None else max_iterations

        while idx < target_max_iterations:
            if not self._is_model_on_gpu:
                self.load_model_to_gpu()

            batch_wait_started_at = time.perf_counter()
            (
                x_batch_inputs,
                x_rewards,
                batch_mean_rewards,
                batch_std_rewards,
                loss_mask,
                ignore_samples,
                server_info,
                batch_metrics,
            ) = self.sample_batch()
            batch_wait_seconds = max(time.perf_counter() - batch_wait_started_at, 0.0)
            group_size = server_info["group_size"]
            micro_batch_size = self._resolve_group_micro_batch_size(group_size)
            microbatch_share = micro_batch_size / max(group_size, 1)
            self._debug_log(
                "train_batch idx=%s group_size=%s micro_batch_size=%s microbatch_share=%.4f "
                "rows=%s seq_len=%s loss_mask_tokens=%s",
                idx,
                group_size,
                micro_batch_size,
                microbatch_share,
                x_batch_inputs.shape[0],
                x_batch_inputs.shape[1] if x_batch_inputs.ndim >= 2 else -1,
                int(loss_mask.sum().item()) if torch.is_tensor(loss_mask) else -1,
            )
            self._debug_gpu_memory(
                "train_batch_loaded",
                idx=idx,
                group_size=group_size,
                micro_batch_size=micro_batch_size,
                microbatch_share=f"{microbatch_share:.4f}",
            )

            expected_rows = self.batch_size * group_size
            if x_batch_inputs.shape[0] != expected_rows:
                self.logger.warning(
                    "Skipping malformed batch: expected %s rows (batch_size=%s, group_size=%s), got %s",
                    expected_rows,
                    self.batch_size,
                    group_size,
                    x_batch_inputs.shape[0],
                )
                continue

            batch_mean_rewards = batch_mean_rewards.unsqueeze(-1).repeat_interleave(group_size, dim=-1)
            batch_std_rewards = batch_std_rewards.unsqueeze(-1).repeat_interleave(group_size, dim=-1)

            batch_inputs = x_batch_inputs.reshape(
                self.batch_size, group_size, *x_batch_inputs.shape[1:]
            )
            loss_mask = loss_mask.reshape(
                self.batch_size, group_size, *loss_mask.shape[1:]
            )
            ignore_samples = ignore_samples.reshape(
                self.batch_size, group_size
            )
            old_policy_log_probs = server_info.get("old_policy_log_probs")
            if old_policy_log_probs is not None:
                pi_old = list(
                    old_policy_log_probs.reshape(
                        self.batch_size, group_size, *old_policy_log_probs.shape[1:]
                    )
                )
            else:
                pi_old = self._compute_old_policy_log_probs(batch_inputs, micro_batch_size)

            iteration_metrics = {
                "total_reward": [],
                "mean_group_reward": [],
                "loss": [],
                "valid_samples": [],
                "kld": [],
            }

            for (
                b_inputs,
                b_old_policy_log_probs,
                b_reward,
                b_loss_mask,
                b_ignore_sample,
                b_mean_rewards,
                b_std_rewards,
            ) in zip(
                batch_inputs, pi_old, x_rewards, loss_mask, ignore_samples, batch_mean_rewards, batch_std_rewards
            ):
                if idx >= target_max_iterations:
                    break

                idx += 1
                group_metrics = self._train_single_group(
                    b_inputs,
                    b_old_policy_log_probs,
                    b_reward,
                    b_loss_mask,
                    b_ignore_sample,
                    b_mean_rewards,
                    b_std_rewards,
                    micro_batch_size,
                )
                step_metrics = {
                    "train/loss": group_metrics["loss"],
                    "train/kld": group_metrics["kld"],
                    "train/total_reward": group_metrics["total_reward"],
                    "train/mean_group_reward": group_metrics["mean_group_reward"],
                    "train/valid_samples": group_metrics["valid_samples"],
                    "perf/group_size": float(group_size),
                    "perf/data_wait_seconds": batch_wait_seconds,
                    "perf/loss_mask_tokens": float(b_loss_mask.sum().item()),
                }
                for key, value in batch_metrics.items():
                    step_metrics[key] = value
                for key, value in group_metrics.items():
                    if "/" in key:
                        step_metrics[key] = value
                step_metrics["train/iteration"] = float(idx)
                self._record_step_metrics(idx, step_metrics)

                print(
                    f"{idx:04d} loss: {group_metrics['loss']:.6f} "
                    f"reward: {group_metrics['total_reward']:.6f} "
                    f"kld: {group_metrics['kld']:.6f} "
                    f"lr: {group_metrics['optim/learning_rate']:.3e} "
                    f"grad_norm: {group_metrics['optim/grad_norm']:.4f} "
                    f"tok/s: {group_metrics['perf/throughput_tokens_per_sec']:.1f} "
                    f"gpu_peak_res_gb: {group_metrics['gpu/peak_reserved_gb']:.2f}"
                )
                self.logger.info(
                    "Step %s | loss %.6f | reward %.6f | kld %.6f | lr %.3e | grad_norm %.4f | "
                    "step_time %.3fs | tok/s %.1f | peak_res %.2fGB",
                    idx,
                    group_metrics["loss"],
                    group_metrics["total_reward"],
                    group_metrics["kld"],
                    group_metrics["optim/learning_rate"],
                    group_metrics["optim/grad_norm"],
                    group_metrics["perf/step_seconds"],
                    group_metrics["perf/throughput_tokens_per_sec"],
                    group_metrics["gpu/peak_reserved_gb"],
                )

                for metric_name, metric_value in group_metrics.items():
                    iteration_metrics[metric_name].append(metric_value)

            if not iteration_metrics["loss"]:
                continue

            avg_iter_reward = sum(iteration_metrics["mean_group_reward"]) / len(iteration_metrics["mean_group_reward"])
            avg_iter_loss = sum(iteration_metrics["loss"]) / len(iteration_metrics["loss"])
            avg_iter_kld = sum(iteration_metrics["kld"]) / len(iteration_metrics["kld"])
            print(f"iter {idx}  >>> reward: {avg_iter_reward}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.logger.info(
                "Iteration summary up to step %s | mean_reward %.6f | mean_loss %.6f | mean_kld %.6f | "
                "data_wait %.3fs",
                idx,
                avg_iter_reward,
                avg_iter_loss,
                avg_iter_kld,
                batch_wait_seconds,
            )
            self._log_to_wandb(
                {
                    "iter/mean_reward": avg_iter_reward,
                    "iter/mean_loss": avg_iter_loss,
                    "iter/mean_kld": avg_iter_kld,
                    "iter/data_wait_seconds": batch_wait_seconds,
                },
                step=idx,
            )

            if self.num_policy_updates > 0 and idx % self.num_policy_updates == 0:
                self.handle_policy_update()
            
            if self.config.save_every > 0 and idx % self.config.save_every == 0:
                self.tokenizer.save_pretrained(f"{self.lora_path}/{idx}")
                self.model.save_pretrained(f"{self.lora_path}/{idx}", adapter_name="grpo")
