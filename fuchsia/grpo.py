import contextlib
import datetime
import gc
import logging
import time
from collections import defaultdict
from typing import Callable, List, Optional, Union, Dict, Any, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from fuchsia.vllm_client import VLLMClient

try:
    from torch.utils.viz._cycles import warn_tensor_cycles
    CYCLE_DETECTION_AVAILABLE = True
except ImportError:
    CYCLE_DETECTION_AVAILABLE = False

from fuchsia.grpo_config import GRPOConfig

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
    ) -> None:
        self.config = config
        self.logger = getattr(config, "logger", logging.getLogger("GRPO"))
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
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
        self.data_loader_iter = iter(self.dataset)
        self.group_size = config.group_size
        self.micro_batch = config.batch_size // config.grad_accumulation_steps
        self.batch_size = config.batch_size
        self.max_iterations = config.max_iterations
        self.dtype = config.dtype
        self.beta = config.beta
        self.epsilon = config.epsilon
        self.epsilon_high = config.epsilon_high
        self.single_gpu = config.single_gpu
        self.non_blocking = False # config.non_blocking
        
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
        self.dataset_feild = config.dataset_feild
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

        self.log_wandb = config.log_wandb
        if self.log_wandb:
            wandb.init(project=config.wandb_project)

        self.metrics = defaultdict(list)
        
        self.model.to(self.dtype)
        if self.beta > 0 and ref_model is not None:
            self.ref_model = self.model


        # Set up proper memory management
        self._setup_memory_management()

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

    def _setup_scheduler(self, config: GRPOConfig) -> None:
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
        
        # wake up cuda allocator
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
        time.sleep(1)
        
        offload_time = time.perf_counter() - offload_start_time
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        self.logger.info(f"CPU offload completed in {offload_time:.2f}s - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        return self.model
    
    @torch.no_grad()
    def load_model_to_gpu(self) -> PreTrainedModel:
        gpu_load_start_time = time.perf_counter()
        self.logger.info("Starting model load to GPU...")
        
        # wake up cuda allocator
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
        time.sleep(1)
        
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
        self.logger.info("Cleaning and syncing memory...")
        torch.cuda.synchronize()
        torch.randn(1).cuda()
        gc.collect()
        torch.cuda.empty_cache()

    def selective_log_softmax(self, logits: Tensor, index: Tensor) -> Tensor:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def get_per_token_logps(self, model: PreTrainedModel, input_ids: Tensor, training: bool = False) -> Tensor:
        logits = model(input_ids=input_ids, training=training).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        logps = self.selective_log_softmax(logits, input_ids)
        return logps

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
        
        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())

        unclipped_loss = policy_ratio * advantage
        clipped_loss = (
            torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon_high) * advantage
        )
        loss = -torch.min(unclipped_loss, clipped_loss)
        
        if self.config.loss_type == "cispo":
            loss = -clipped_loss.detach() * policy_log_probs
            loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
            return loss.mean(), 0.0  # Return 0.0 for KLD when using CISPO
        
        if self.ignore_imcomplete_samples:
            loss = loss * ignore_sample
            kld = kld * ignore_sample
        
        loss = (loss * loss_mask).sum(dim=-1) / (torch.ones_like(loss_mask).sum(dim=-1) + 1e-6) # dr grpo loss
        kld =  (kld  * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)

        loss += kld * self.beta

        # Calculate average KLD for this batch
        avg_kld = kld.mean().item() if self.beta != 0 else 0.0

        return loss.mean(), avg_kld

    def sample_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        inputs_texts = []
        outputs = []
        completions = []
        rewards = []
        mean_rewards = []
        std_rewards = []
        ignore_sample = []
        
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            
            rewards.append(item["rewards"])
            mean_rewards.append(item["mean"])
            std_rewards.append(item["std"])
            
            for idx in range(len(item["completions"])):
                prompt = item["inputs"]
                inputs_texts.append(prompt)
                completions.append(item["completions"][idx])
                ignore_sample.append(item["finish_reason"][idx] != "length")
                
        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]
        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        outputs = []
        self.metrics["samples"].append({"prompt": decoded, "completions": completions})
        
        avg_token_lengths = 0
        for completion in completions:
            avg_token_lengths += len(self.tokenizer.encode(completion))
        avg_token_lengths /= len(completions)
        
        self.metrics["avg_token_lengths"].append(avg_token_lengths)

        for prompt, completion in zip(decoded, completions):
            outputs.append(prompt + completion)

        outputs = self.tokenizer(
            outputs, padding=True, padding_side="right", return_tensors="pt"
        )["input_ids"]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)

        decoded_outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=False
        )
        
        if self.config.debug:   
            print("\n\n\n")
            print("-" * 10)
            print(decoded_outputs[0].replace("<|finetune_right_pad_id|>", "").replace("<|end_of_text|>", ""))
            print("-" * 10)
            print("\n\n\n")
            
        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)
        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return (
            outputs,
            torch.tensor(rewards, dtype=self.dtype).float(),
            torch.tensor(mean_rewards, dtype=self.dtype).float(),
            torch.tensor(std_rewards, dtype=self.dtype).float(),
            loss_mask[:, 1:],
            torch.tensor(ignore_sample, dtype=torch.bool).to(torch.int8)
        )

    def log_metrics(self) -> None:
        if not self.log_wandb:
            return
            
        current_idx = self.metrics.get("idx", [0])[-1] if self.metrics.get("idx") else 0
        
        metrics = {
            f"train/{k}": v[-1] 
            for k, v in self.metrics.items() 
            if k != "idx" and v
        }
        metrics["train/iteration"] = current_idx
        
        wandb.log(metrics)

    def handle_policy_update(self) -> None:
        self.vllm_client.update_model_params(
            self.model, lora=self.using_lora, 
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
        """Perform VLLM hotswap cycle: offload model, wake VLLM, fill buffer, wait for memory, load model back."""
        vllm_cycle_start_time = time.perf_counter()
        self.logger.info("Starting VLLM hotswap cycle...")
        
        self.offload_to_cpu()
        vllm_wake_start_time = time.perf_counter()
        self.vllm_client.wake_up()
        self.vllm_client.fill_buffer()
        self.vllm_client.sleep()
        
        # Wait for GPU memory to drop below half capacity
        while True:
            allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_memory_gb = torch.cuda.memory_reserved() / (1024**3)
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if allocated_memory_gb < total_memory_gb / 2:
                self.logger.info(f"GPU memory below half capacity: {allocated_memory_gb:.2f}GB / {total_memory_gb:.2f}GB")
                break
            
            self.logger.info(f"Waiting for GPU memory to drop below half capacity: {allocated_memory_gb:.2f}GB / {total_memory_gb:.2f}GB")
            time.sleep(2)
            
        vllm_wake_time = time.perf_counter() - vllm_wake_start_time
        self.load_model_to_gpu()
        total_vllm_cycle_time = time.perf_counter() - vllm_cycle_start_time
        self.logger.info(f"VLLM hotswap cycle completed in {total_vllm_cycle_time:.2f}s (VLLM wake/fill/sleep: {vllm_wake_time:.2f}s)")

    def train(self, epochs: int = 1, max_iterations: int = 10000) -> None:
        
        idx = 0
        start_time = time.perf_counter()
        
        while idx < max_iterations:
            if not self._is_model_on_gpu:
                self.load_model_to_gpu()
            
            x_batch_inputs, x_rewards, batch_mean_rewards, batch_std_rewards, loss_mask, ignore_samples = self.sample_batch()
            
            batch_mean_rewards = batch_mean_rewards.unsqueeze(-1).repeat_interleave(self.group_size, dim=-1)
            batch_std_rewards  = batch_std_rewards.unsqueeze(-1).repeat_interleave(self.group_size, dim=-1)
            
            batch_inputs = x_batch_inputs.reshape(
                self.batch_size, self.group_size, *x_batch_inputs.shape[1:]
            ).cpu()
            loss_mask = loss_mask.reshape(
                self.batch_size, self.group_size, *loss_mask.shape[1:]
            ).cpu()
            ignore_samples = ignore_samples.reshape(
                self.batch_size, self.group_size
            ).cpu()
            x_rewards = x_rewards.cpu()

            pi_old = []
            with torch.no_grad():
                for b_inputs in batch_inputs:
                    x_old_policy_log_probs = self.get_per_token_logps(
                        self.model, b_inputs.to(self.device)
                    ).cpu()
                    pi_old.append(x_old_policy_log_probs)
                torch.cuda.empty_cache()

            self.optimizer.zero_grad()
            
            # Collect metrics for this outer loop iteration
            iteration_metrics = {
                "total_reward": [],
                "mean_group_reward": [],
                "loss": [],
                "valid_samples": [],
                "kld": []
            }
            
            for b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask, b_ignore_sample, b_mean_rewards, b_std_rewards in zip(
                batch_inputs, pi_old, x_rewards, loss_mask, ignore_samples, batch_mean_rewards, batch_std_rewards
            ):
                idx += 1
                group_losses = []
                group_klds = []
                
                reward = b_reward.to(self.device)

                for start in range(0, b_inputs.shape[0], self.micro_batch):
                    end = start + self.micro_batch
                    inputs = b_inputs[start:end]
                    old_policy_log_probs = b_old_policy_log_probs[start:end]
                    reward_batch = b_reward[start:end]
                    loss_mask_batch = b_loss_mask[start:end]
                    ignore_sample_batch = b_ignore_sample[start:end]
                    mean_rewards = b_mean_rewards[start:end]
                    std_rewards = b_std_rewards[start:end]
                    
                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward_batch = reward_batch.to(self.device)
                    loss_mask_batch = loss_mask_batch.to(self.device)
                    ignore_sample_batch = ignore_sample_batch.unsqueeze(-1).to(self.device)
                    
                    mean_rewards = mean_rewards.to(self.device)
                    std_rewards = std_rewards.to(self.device)

                    loss, kld = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward_batch,
                        mean_rewards,
                        std_rewards,
                        loss_mask_batch,
                        ignore_sample_batch
                    )
                    
                    group_losses.append(loss.item())
                    group_klds.append(kld)
                    loss.backward()

                print(f"{idx:04d} loss: {sum(group_losses) / len(group_losses)} reward: {reward.mean()}")
                
                # Collect metrics for this group
                iteration_metrics["total_reward"].append(reward.mean().item())
                iteration_metrics["mean_group_reward"].append(batch_mean_rewards.mean().item())
                iteration_metrics["loss"].append(sum(group_losses) / len(group_losses))
                iteration_metrics["valid_samples"].append(b_ignore_sample.sum().item())
                iteration_metrics["kld"].append(sum(group_klds) / len(group_klds))
            
            # Step the optimizer
            self.optimizer.step()

            # Step the learning rate scheduler if enabled
            if self.use_scheduler and self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                if self.log_wandb:
                    self.metrics["learning_rate"].append(current_lr)
                self.logger.debug(f"Current learning rate: {current_lr:.2e}")

            # Log metrics for this iteration (averaged across all groups)
            if self.log_wandb:
                self.metrics["idx"].append(idx)
                self.metrics["total_reward"].append(sum(iteration_metrics["total_reward"]) / len(iteration_metrics["total_reward"]))
                self.metrics["mean_group_reward"].append(sum(iteration_metrics["mean_group_reward"]) / len(iteration_metrics["mean_group_reward"]))
                self.metrics["loss"].append(sum(iteration_metrics["loss"]) / len(iteration_metrics["loss"]))
                self.metrics["valid_samples"].append(sum(iteration_metrics["valid_samples"]))
                self.metrics["kld"].append(sum(iteration_metrics["kld"]) / len(iteration_metrics["kld"]))

            print(f"iter {idx}  >>> reward: {batch_mean_rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.log_metrics()

            if idx % self.num_policy_updates == 0:
                self.handle_policy_update()
            
            if idx % self.config.save_every == 0:
                self.model.save_pretrained(f"{self.lora_path}/{idx}", adapter_name="grpo")

