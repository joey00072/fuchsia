import torch
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time
import wandb
import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Callable, Any

from fuchsia.vllm_client import VLLMClient


@dataclass
class GRPOConfig:
    group_size: int = 8
    micro_group_size: int = 2
    batch_size: int = 1
    max_iterations: int = 1000
    log_wandb: bool = False
    dtype: str = "bfloat16"
    lr: float = 5e-6
    weight_decay: float = 0.0
    beta: float = 0.0
    epsilon: float = 0.2
    using_lora: bool = False
    wandb_project: str = "fuchsia"
    use_vllm: bool = False
    dataset_feild: str = "prompt"
    num_policy_updates: int = 8

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


class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        dataset,
        optimizer=None,
        reward_functions: List[Callable] = None,
        config: GRPOConfig = None,
        vllm_client: VLLMClient = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(model)
        self.config = config

        self.model = model
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
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(
                self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        )
        self.reward_functions: list = reward_functions
        self.dataset_feild = config.dataset_feild
        self.num_policy_updates = config.num_policy_updates

        self.using_lora = config.using_lora
        if self.using_lora and self.beta > 0:
            self.ref_model = model

        self.distributed = config.use_vllm
        self.log_wandb = config.log_wandb
        if self.log_wandb:
            wandb.init(project=config.wandb_project)

        self.metrics = defaultdict(list)

        self.model.to(self.device).to(config.dtype)
        if self.beta > 0 and ref_model is not None:
            self.ref_model.to(self.device).to(config.dtype)

        if config.use_vllm:
            print("Using VLLM")
            self.vllm_client = vllm_client if vllm_client is not None else VLLMClient()

    def selective_log_softmax(self, logits, index):
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = self.selective_log_softmax(logits, input_ids)
        return logps

    def compute_loss(
        self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask
    ) -> Tensor:
        policy_log_probs = self.get_per_token_logps(self.model, inputs)

        kld = 0
        if self.beta != 0:
            with (
                self.ref_model.disable_adapter()
                if self.using_lora
                else contextlib.nullcontext()
            ):
                ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)
            # kl divergence calculation
            log_ratios = ref_policy_log_probs - policy_log_probs
            kld = torch.exp(log_ratios) - log_ratios - 1

        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)

        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())

        loss1 = policy_ratio * advantage
        loss2 = (
            torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        )
        loss = -torch.min(loss1, loss2)
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-6)

        loss += kld * self.beta

        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())

        return loss.mean()

    def sample_batch(self):
        if self.distributed:
            return self.distributed_sample_batch()

        inputs_texts = []
        samples = []
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(prompt, tokenize=False)
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        max_new_tokens = 512
        outputs = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            repetition_penalty=1.1,
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=False
        )


    
        rewards = self.compute_rewards(samples, decoded_outputs)

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return (
            outputs,
            torch.tensor(rewards, dtype=self.dtype).float(),
            loss_mask[:, 1:],
        )

    def distributed_sample_batch(self):
        inputs_texts = []
        outputs = []
        completions = []
        samples = []
        rewards = []
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            rewards.append(item["rewards"])
            for idx in range(len(item["completions"])):
                samples.append(item)
                prompt = item["inputs"]
                inputs_texts.append(prompt)
                completions.append(item["completions"][idx])

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        outputs = []

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
        
        print("\n\n\n")
        print("-" * 10)
        print(decoded_outputs[0].replace("<|finetune_right_pad_id|>", ""))
        print("-" * 10)
        print("\n\n\n")
        
        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return (
            outputs,
            torch.tensor(rewards, dtype=self.dtype).float(),
            loss_mask[:, 1:],
        )

    def compute_rewards(self, samples, responses):
        rewards = [
            [[] for _ in range(self.batch_size)]
            for _ in range(len(self.reward_functions))
        ]

        for idx, (sample, resp) in enumerate(zip(samples, responses)):
            for func_idx, func in enumerate(self.reward_functions):
                reward = func(sample, resp)
                rewards[func_idx][idx % self.batch_size].append(reward)

        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)

        for func_idx, func in enumerate(self.reward_functions):
            rwds = rewards[func_idx].mean(dim=-1)
            for r in rwds:
                self.metrics[f"reward_{func.__name__}"].append(r.item())

        prompt_lenghts = [[] for _ in range(self.batch_size)]
        for idx, resp in enumerate(responses):
            prompt_lenghts[idx % self.batch_size].append(len(resp))

        for idx, pl in enumerate(prompt_lenghts):
            self.metrics[f"prompt_length"].append(sum(pl) / len(pl))

        return rewards.sum(dim=0)

    def log_metrics(self):
        if self.log_wandb:
            idx = self.metrics["idx"][-1] - 1
            metrics = {}
            for k, v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]

            wandb.log(metrics)

    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        start_time = time.perf_counter()
        while idx < max_iterations:
            x_batch_inputs, rewards, loss_mask = self.sample_batch()

            batch_inputs = x_batch_inputs.reshape(
                self.batch_size, self.group_size, *x_batch_inputs.shape[1:]
            )
            loss_mask = loss_mask.reshape(
                self.batch_size, self.group_size, *loss_mask.shape[1:]
            )

            # offload to cpu to save vram
            batch_inputs = batch_inputs.cpu()
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()

            pi_old = []
            for _, (b_inputs) in enumerate(batch_inputs):
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logps(
                        self.model, b_inputs.to(self.device)
                    ).cpu()
                    pi_old.append(b_old_policy_log_probs)

            for _, (
                b_inputs,
                b_old_policy_log_probs,
                b_reward,
                b_loss_mask,
            ) in enumerate(zip(batch_inputs, pi_old, rewards, loss_mask)):
                idx += 1
                reward = b_reward.to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)

                # even grop are too big for vram
                # so we split them into micro groups (its same as micro batching)
                g_inputs = b_inputs.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_inputs.shape[1:],
                ).cpu()
                g_old_policy_log_probs = b_old_policy_log_probs.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_old_policy_log_probs.shape[1:],
                ).cpu()
                g_reward = b_reward.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_reward.shape[1:],
                ).cpu()
                g_loss_mask = b_loss_mask.reshape(
                    b_inputs.shape[0] // self.micro_group_size,
                    self.micro_group_size,
                    *b_loss_mask.shape[1:],
                ).cpu()
                group_losses = []

                for inputs, old_policy_log_probs, reward, loss_mask in zip(
                    g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask
                ):
                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward = reward.to(self.device)
                    loss_mask = loss_mask.to(self.device)

                    loss = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward,
                        mean_rewards,
                        std_rewards,
                        loss_mask,
                    )
                    group_losses.append(loss.item())
                    loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(
                    f"{idx:04d} loss: {sum(group_losses) / len(group_losses)} reward: {reward.mean()}"
                )
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["loss"].append(sum(group_losses) / len(group_losses))

            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(
                f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}"
            )
            self.log_metrics()

            if idx % self.num_policy_updates == 0:
                self.vllm_client.update_model_params(self.model)
                self.vllm_client.empty_buffer()
                self.vllm_client.fill_buffer()
