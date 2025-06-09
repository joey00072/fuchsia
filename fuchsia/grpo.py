import contextlib
import datetime
import gc
import logging
import time
from collections import defaultdict
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from peft import PeftModel
from torch import Tensor
from transformers import AutoModelForCausalLM

try:
    from torch.utils.viz._cycles import warn_tensor_cycles
    CYCLE_DETECTION_AVAILABLE = True
except ImportError:
    CYCLE_DETECTION_AVAILABLE = False

class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        dataset,
        optimizer=None,
        reward_functions: List[Callable] = None,
        config=None,
        vllm_client=None,
    ):
        self.config = config
        self.logger = getattr(config, "logger", logging.getLogger("GRPO"))
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
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
        self.micro_group_size = config.micro_group_size
        self.batch_size = config.batch_size
        self.max_iterations = config.max_iterations
        self.dtype = config.dtype
        self.beta = config.beta
        self.epsilon = config.epsilon
        self.epsilon_high = config.epsilon_high
        self.single_gpu = config.single_gpu
        
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                self.model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay
            )
        )
        
        self.logger.info(f"Learning rate: {config.lr}")
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
        
        if not config.use_vllm and self.ignore_imcomplete_samples:
            self.logger.warning("ignore_imcomplete_samples is set to True, but use_vllm is set to False. This will not have any effect.")
            
        if self.using_lora and self.beta > 0:
            self.ref_model = model

        self.distributed = config.use_vllm
        self.log_wandb = config.log_wandb
        if self.log_wandb:
            wandb.init(project=config.wandb_project)

        self.metrics = defaultdict(list)
        
        self.model.to(self.dtype)
        if self.beta > 0 and ref_model is not None:
            self.ref_model = self.model

        if config.use_vllm:
            self.logger.info("Using VLLM")
            self.vllm_client = vllm_client if vllm_client is not None else VLLMClient()
            
        # if self.single_gpu:
        #     self.model.save_pretrained(self.lora_path)

        # Set up proper memory management
        self._setup_memory_management()

    def _setup_memory_management(self):
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



    @torch.no_grad()
    def offload_to_cpu(self):
        """Improved CPU offloading with proper memory management."""
        self.logger.info("Starting offload to CPU...")
        
        # wake up cuda allocator
        torch.randn(1).cuda()
        
        for param in self.model.parameters():
            param.data = param.data.to("cpu", non_blocking=True)
            if hasattr(param, "_local_shard"): # this need for fsdp
                param._local_shard = param.data
            if param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
                
        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cpu", non_blocking=True)
            
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu", non_blocking=True)
        
        self.model.eval()        
        self._is_model_on_gpu = False
        self._is_optimizer_on_gpu = False
        
        self.clean_and_sync_memory()
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        self.logger.info(f"GPU Memory after offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return self.model
    
    @torch.no_grad()
    def load_model_to_gpu(self):
        self.logger.info("Starting model load to GPU...")
        
        # wake up cuda allocator
        torch.randn(1).cuda()
        
        for param in self.model.parameters():
            param.data = param.data.to("cuda", non_blocking=True)
            
            if hasattr(param, "_local_shard"): # this need for fsdp
                param._local_shard = param.data
            if param.grad is not None:
                param.grad = param.grad.to("cuda", non_blocking=True)
                
        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cuda", non_blocking=True)
            
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to("cuda", non_blocking=True)
    
        self._is_model_on_gpu = True
        self._is_optimizer_on_gpu = True
        
        self.clean_and_sync_memory()
        
        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_memory_gb = torch.cuda.memory_reserved() / (1024**3)
        self.logger.info(f"GPU Memory after load: {allocated_memory_gb:.2f}GB allocated, {reserved_memory_gb:.2f}GB reserved")
        
        return self.model

    def prepare_for_inference(self):
        if not self._is_model_on_gpu:
            self.load_model_to_gpu()
        self.model.eval()
        self.move_optimizer_to_cpu()
        

    def prepare_for_training(self):
        if not self._is_model_on_gpu:
            self.load_model_to_gpu()
        self.model.train()

    @torch.no_grad()
    def move_optimizer_to_cpu(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to("cpu", non_blocking=True)
        
        self._is_optimizer_on_gpu = False
        self.clean_and_sync_memory()

    @torch.no_grad()
    def cleanup_tensors(self, *tensors):
        for tensor in tensors:
            if hasattr(tensor, 'data'):
                del tensor
        self.clean_and_sync_memory()
        
    def clean_and_sync_memory(self):
        self.logger.info("Cleaning and syncing memory...")
        torch.cuda.synchronize()
        torch.randn(1).cuda()
        gc.collect()
        torch.cuda.empty_cache()

    def selective_log_softmax(self, logits, index):
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def get_per_token_logps(self, model, input_ids, training=False) -> Tensor:
        logits = model(input_ids=input_ids, training=training).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        logps = self.selective_log_softmax(logits, input_ids)
        return logps

    def compute_loss(
        self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask, ignore_sample
    ) -> Tensor:
        
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
        
        if self.ignore_imcomplete_samples:
            loss = loss * ignore_sample
            kld = kld * ignore_sample
        
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        kld =  (kld  * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)

        loss += kld * self.beta

        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())

        return loss.mean()

    def sample_batch(self):
        inputs_texts = []
        outputs = []
        completions = []
        samples = []
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
                samples.append(item)

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
        samples = [sample for _ in range(self.group_size) for sample in samples]

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

        # del encoded, attention_mask, decoded, gen_tokens, valid_gen_mask
        # self.clean_and_sync_memory()

        return (
            outputs,
            torch.tensor(rewards, dtype=self.dtype).float(),
            torch.tensor(mean_rewards, dtype=self.dtype).float(),
            torch.tensor(std_rewards, dtype=self.dtype).float(),
            loss_mask[:, 1:],
            torch.tensor(ignore_sample, dtype=torch.bool).to(torch.int8)
        )

    def log_metrics(self):
        if self.log_wandb:
            idx = self.metrics["idx"][-1] - 1
            metrics = {}
            for k, v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]
            wandb.log(metrics)


    def train(self, epochs=1, max_iterations=10000):
        
        idx = 0
        start_time = time.perf_counter()
        
        while idx < max_iterations:
            if not self._is_model_on_gpu:
                self.load_model_to_gpu()
            
            x_batch_inputs, x_rewards, batch_mean_rewards, batch_std_rewards, loss_mask, ignore_samples = self.sample_batch()
            
            
            batch_mean_rewards = batch_mean_rewards.unsqueeze(-1).repeat_interleave(self.group_size, dim=-1)
            batch_std_rewards = batch_std_rewards.unsqueeze(-1).repeat_interleave(self.group_size, dim=-1)
            
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

            # self.cleanup_tensors(x_batch_inputs)

            pi_old = []
            with torch.no_grad():
                for b_inputs in batch_inputs:
                    x_old_policy_log_probs = self.get_per_token_logps(
                        self.model, b_inputs.to(self.device)
                    ).cpu()
                    pi_old.append(x_old_policy_log_probs)
                torch.cuda.empty_cache()

            for b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask, b_ignore_sample, b_mean_rewards, b_std_rewards in zip(
                batch_inputs, pi_old, x_rewards, loss_mask, ignore_samples, batch_mean_rewards, batch_std_rewards
            ):
                idx += 1
                
                reward = b_reward.to(self.device)
                mean_x_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_x_rewards = reward.std(dim=-1).unsqueeze(-1)


                g_inputs = b_inputs.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_inputs.shape[1:],
                )
                g_old_policy_log_probs = b_old_policy_log_probs.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_old_policy_log_probs.shape[1:],
                )
                g_reward = b_reward.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_reward.shape[1:],
                )
                g_loss_mask = b_loss_mask.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_loss_mask.shape[1:],
                )
                g_ignore_sample = b_ignore_sample.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_ignore_sample.shape[1:],
                )
                
                g_mean_rewards = b_mean_rewards.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_mean_rewards.shape[1:],
                )
                g_std_rewards = b_std_rewards.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_std_rewards.shape[1:],
                )
                
                group_losses = []
                self.optimizer.zero_grad(set_to_none=True)
                
                

                for inputs, old_policy_log_probs, reward_batch, loss_mask_batch, ignore_sample_batch, mean_rewards, std_rewards in zip(
                    g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask, g_ignore_sample, g_mean_rewards, g_std_rewards
                ):
                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward_batch = reward_batch.to(self.device)
                    loss_mask_batch = loss_mask_batch.to(self.device)
                    ignore_sample_batch = ignore_sample_batch.unsqueeze(-1).to(self.device)
                    
                    mean_rewards = mean_rewards.to(self.device)
                    std_rewards = std_rewards.to(self.device)

                    loss = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward_batch,
                        mean_rewards,
                        std_rewards,
                        loss_mask_batch,
                        ignore_sample_batch
                    )
                    
                    group_losses.append(loss.item())
                    loss.backward()
                    
                    # self.cleanup_tensors(
                    #     inputs, old_policy_log_probs, reward_batch, 
                    #     loss_mask_batch, ignore_sample_batch, loss
                    # )

                print(f"{idx:04d} loss: {sum(group_losses) / len(group_losses)} reward: {reward.mean()}")
                
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["mean_group_reward"].append(batch_mean_rewards.mean().item())
                    self.metrics["loss"].append(sum(group_losses) / len(group_losses))
                    self.metrics["valid_samples"].append(b_ignore_sample.sum().item())
            self.optimizer.step()

            # del batch_inputs, loss_mask, ignore_samples, x_rewards, pi_old, loss, x_batch_inputs, reward, mean_x_rewards, std_x_rewards
            # self.clean_and_sync_memory()

            print(f"iter {idx}  >>> reward: {batch_mean_rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.log_metrics()

            if idx % self.num_policy_updates == 0 and self.distributed:
                self.vllm_client.update_model_params(
                    self.model, lora=self.using_lora, 
                    single_gpu=self.single_gpu, lora_path=self.lora_path
                )
                
                if not self.async_buffer_fill:
                    self.vllm_client.empty_buffer()
                    
                if self.single_gpu:
                    self.logger.info("Offloading model to CPU")
                    self.offload_to_cpu()
                    time.sleep(1)
                    
                    self.logger.info("Waking up VLLM client")
                    self.vllm_client.wake_up()
                    time.sleep(1)
                    
                self.logger.info("Filling buffer")
                self.vllm_client.fill_buffer()
                # time.sleep(5)
                
                if self.single_gpu:
                    self.logger.info("Putting VLLM client to sleep")
                    self.vllm_client.sleep()
                    time.sleep(1)
                    self.logger.info("Loading model back to GPU")
                    self.load_model_to_gpu()
            
            if idx % self.config.save_every == 0:
                self.model.save_pretrained(f"{self.lora_path}/{idx}", adapter_name="grpo")

