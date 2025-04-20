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

from lightning.fabric import Fabric
from fuchsia.vllm_client import VLLMClient
from liger_kernel.transformers import AutoLigerKernelForCausalLM

import bitsandbytes as bnb
from lightning.fabric.strategies import FSDPStrategy

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed._composable.fsdp.fully_shard import fully_shard

from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.loggers import TensorBoardLogger

policy = partial(size_based_auto_wrap_policy, min_num_params=10000)
activation_checkpointing_policy={
        LlamaDecoderLayer,
    }   
strategy = FSDPStrategy(
    # Full-shard within a machine, replicate across machines
    sharding_strategy="HYBRID_SHARD",
    auto_wrap_policy=policy,
    activation_checkpointing_policy=activation_checkpointing_policy,
    state_dict_type="sharded",
)


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
    epsilon: float = 0.1
    using_lora: bool = False
    wandb_project: str = "nanoGRPO"
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
        self.config = config
        wandb_logger = WandbLogger(project=config.wandb_project)
        self.fabric = Fabric(accelerator="cuda", precision="bf16-true",devices=2, strategy=strategy, loggers=[wandb_logger])
        self.fabric.launch()
        self.device = self.fabric.device

        self.model = AutoModelForCausalLM.from_pretrained(model).to(torch.bfloat16)
        print(self.model)
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(
                self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
        )
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.epoch = 1
        
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

        self._enable_gradient_checkpointing(self.model)
        self.reward_functions: list = reward_functions
        self.dataset_feild = config.dataset_feild
        self.num_policy_updates = config.num_policy_updates

        self.using_lora = config.using_lora
        if self.using_lora and self.beta > 0:
            self.ref_model = model

        self.distributed = config.use_vllm
        self.log_wandb = config.log_wandb
        # if self.log_wandb:
        #     wandb.init(project=config.wandb_project)

        self.metrics = defaultdict(list)
        
        if self.beta > 0 and ref_model is not None and not self.using_lora:
            self.ref_model = self.fabric.setup_module(self.ref_model)

        if config.use_vllm:
            print("Using VLLM")
            self.vllm_client = vllm_client if vllm_client is not None else VLLMClient()

    def _enable_gradient_checkpointing(self, model):
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def selective_log_softmax(self, logits, index):
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        return torch.stack(per_token_logps)

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        return self.selective_log_softmax(logits, input_ids)

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        policy_log_probs = self.get_per_token_logps(self.model, inputs)

        kld = 0
        if self.beta != 0:
            with (
                self.ref_model.disable_adapter()
                if self.using_lora
                else contextlib.nullcontext()
            ):
                ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)
            log_ratios = ref_policy_log_probs - policy_log_probs
            kld = torch.exp(log_ratios) - log_ratios - 1

        advantage = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)

        policy_ratio = torch.exp(policy_log_probs - old_policy_log_probs.detach())

        loss1 = policy_ratio * advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
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
            input_ids.to(self.fabric.device),
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            repetition_penalty=1.1,
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=False
        )

        import rich

        rich.print(decoded_outputs[0])

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
            self.epoch = item["epoch"]
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

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return (
            outputs,
            torch.tensor(rewards, dtype=self.dtype).float(),
            loss_mask[:, 1:],
        )

    def train(self, epochs=1, max_iterations=1000):
        idx = 0
        start_time = time.perf_counter()
        while idx < max_iterations:
            x_batch_inputs, rewards, loss_mask = self.sample_batch()

            batch_inputs = x_batch_inputs.reshape(self.batch_size, self.group_size, *x_batch_inputs.shape[1:])
            loss_mask = loss_mask.reshape(self.batch_size, self.group_size, *loss_mask.shape[1:])

            batch_inputs = batch_inputs.cpu()
            rewards = rewards.cpu()
            loss_mask = loss_mask.cpu()

            pi_old = []
            for b_inputs in batch_inputs:
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logps(self.model, b_inputs.to(self.fabric.device)).cpu()
                    pi_old.append(b_old_policy_log_probs)

            for b_inputs, b_old_policy_log_probs, b_reward, b_loss_mask in zip(batch_inputs, pi_old, rewards, loss_mask):
                idx += 1
                reward = b_reward.to(self.fabric.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)

                g_inputs = b_inputs.reshape(b_inputs.shape[0] // self.micro_group_size, self.micro_group_size, *b_inputs.shape[1:]).cpu()
                g_old_policy_log_probs = b_old_policy_log_probs.reshape(g_inputs.shape[0], self.micro_group_size, *b_old_policy_log_probs.shape[1:]).cpu()
                g_reward = b_reward.reshape(g_inputs.shape[0], self.micro_group_size, *b_reward.shape[1:]).cpu()
                g_loss_mask = b_loss_mask.reshape(g_inputs.shape[0], self.micro_group_size, *b_loss_mask.shape[1:]).cpu()

                group_losses = []
                for inputs, old_policy_log_probs, reward, loss_mask in zip(g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask):
                    inputs = inputs.to(self.fabric.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.fabric.device)
                    reward = reward.to(self.fabric.device)
                    loss_mask = loss_mask.to(self.fabric.device)

                    loss = self.compute_loss(inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask)
                    group_losses.append(loss.item())
                    self.fabric.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.fabric.print(f"{idx:04d} loss: {sum(group_losses) / len(group_losses)} reward: {reward.mean()}")
                
                # Add fabric logging
                elapsed_time = time.perf_counter() - start_time
                self.fabric.log_dict(
                    {
                        "training_loss": sum(group_losses) / len(group_losses),
                        "reward": reward.mean().item(),
                        "iter": idx,
                        "tokens": idx * self.group_size * self.micro_group_size,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "time": elapsed_time,
                        "epoch": self.epoch,
                    },
                    step=idx,
                )

                # if self.log_wandb:
                #     self.metrics["idx"].append(idx)
                #     self.metrics["total_reward"].append(reward.mean().item())
                #     self.metrics["loss"].append(sum(group_losses) / len(group_losses))

            self.fabric.print(f"iter {idx}  >>> reward: {rewards.mean()}")
            self.fabric.print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            # self.log_metrics()

            if idx % self.num_policy_updates == 0 and self.fabric.is_global_zero:
                if self.fabric.local_rank == 0:
                    self.vllm_client.update_model_params(self.model)
                    self.vllm_client.empty_buffer() 
                    self.vllm_client.fill_buffer()
                    
            if idx % 1 == 0:
                self.fabric.barrier()
                
                print(f"Saving checkpoint at {idx}")
                state = {"model": self.model, "iter": idx}
                self.fabric.save(f"/mnt/nvme0n1/checkpoint/ckpt_{idx}", state)
                print(f"Saved checkpoint at {idx}")




from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
import math
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from fuchsia.dist_dataset import DatasetClient
from fuchsia.vllm_client import VLLMClient


import os


SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[red]Failed to load config from {config_path}: {e}[/red]")
        raise


def prepare_dataset(dataset) -> Dataset:
    """Prepare the GSM8K dataset with better error handling and validation."""

    def extract_hash_answer(text: str) -> Optional[str]:
        try:
            if "####" not in text:
                return None
            answer = text.split("####")[1].strip()
            # Validate that the answer is a number
            float(answer)
            return answer
        except (ValueError, IndexError):
            return None

    def process_example(example: dict) -> Optional[dict]:
        try:
            answer = extract_hash_answer(example["answer"])
            if answer is None:
                return None
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ],
                "answer": answer,
            }
        except Exception as e:
            print(f"[yellow]Failed to process example: {e}[/yellow]")
            return None

    try:
        dataset = dataset.map(
            process_example,
            remove_columns=[
                col for col in dataset.column_names if col not in ["prompt", "answer"]
            ],
            desc="Processing dataset",
        )
        dataset = dataset.filter(lambda x: x is not None)
        print(f"[green]Processed dataset size: {len(dataset)}[/green]")
        return dataset
    except Exception as e:
        print(f"[red]Failed to prepare dataset: {e}[/red]")
        raise


def response_format_reward(sample: dict, s: str, *args, **kwargs) -> float:
    """Improved reward function with better validation and scoring."""
    END_OF_TEXT_TOKEN = "<|eot_id|>"
    START_HEADER_TOKEN = "<|start_header_id|>"
    END_HEADER_TOKEN = "<|end_header_id|>"
    ASSISTANT_TOKEN = "assistant"
    USER_TOKEN = "user"

    START_THINKING_TOKEN = "<thinking>"
    END_THINKING_TOKEN = "</thinking>"
    START_ANSWER_TOKEN = "<answer>"
    END_ANSWER_TOKEN = "</answer>"

    try:
        # Extract the actual response
        try:
            s = s.split(
                f"{END_OF_TEXT_TOKEN}{START_HEADER_TOKEN}{ASSISTANT_TOKEN}{END_HEADER_TOKEN}"
            )[1]
        except IndexError:
            return -1.0

        if END_OF_TEXT_TOKEN in s:
            s = s.split(END_OF_TEXT_TOKEN)[0]

        # Initialize reward components
        format_reward = 0.0
        content_reward = 0.0
        correct_template = 0

        # Check format tags
        required_tags = [
            START_THINKING_TOKEN,
            END_THINKING_TOKEN,
            START_ANSWER_TOKEN,
            END_ANSWER_TOKEN,
        ]
        for tag in required_tags:
            if tag in s:
                format_reward += 0.15
                if s.count(tag) > 1:
                    format_reward -= s.count(tag) * 0.01

        # Validate thinking section
        if s.count("<thinking>") == 1:
            format_reward += 0.5
            thinking_content = (
                s.split(START_THINKING_TOKEN)[1].split(END_THINKING_TOKEN)[0].strip()
            )
            if len(thinking_content) > 10:  # Basic content validation
                content_reward += 0.5
        else:
            format_reward -= 0.1

        # Validate answer section
        if "<answer>" in s and "</answer>" in s:
            format_reward += 0.4
            answer_content = (
                s.split(START_ANSWER_TOKEN)[1].split(END_ANSWER_TOKEN)[0].strip()
            )
            try:
                answer_value = float(answer_content)
                content_reward += 1.0
                if answer_value == float(sample["answer"]):
                    content_reward += 2.0
                    correct_template += 1
            except ValueError:
                content_reward -= 0.1

        # Bonus for perfect format
        if correct_template == 1:
            format_reward += 2.0

        return format_reward + content_reward

    except Exception as e:
        print(f"[yellow]Error in reward calculation: {e}[/yellow]")
        return -1.0


import requests


class MockClient:
    def __init__(self):
        self.url = "http://localhost:8000/"

    def get_sample(self):
        url = self.url + "get_sample/"
        response = requests.get(url)
        return response.json()["sample"]

    def update_model_params(self, model):
        for name, param in model.named_parameters():
            print(name, param)
        pass

    def empty_buffer(self):
        pass

    def fill_buffer(self):
        pass


def main():
    try:
        # Load configuration
        config_path = Path(__file__).parent / "gsm8k_config.yaml"
        config = load_config(str(config_path))

        # Initialize model and tokenizer
        print(f"[blue]Loading model: {config['model_name']}[/blue]")
        # model = AutoModelForCausalLM.from_pretrained(config["model_name"]).to(
        #     torch.bfloat16
        # )
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])


        # Load and prepare dataset
        print("[blue]Loading GSM8K dataset[/blue]")
        # dataset = load_dataset("openai/gsm8k", "main")["train"]
        # dataset = prepare_dataset(dataset)
        # vllm_client = MockClient()
        vllm_client = VLLMClient()
        dataset = DatasetClient(vllm_client)

        # Configure GRPO
        grpo_config = GRPOConfig(
            group_size=config["grpo"]["group_size"],
            micro_group_size=config["grpo"]["micro_group_size"],
            batch_size=config["grpo"]["batch_size"],
            lr=float(config["grpo"]["lr"]),
            weight_decay=float(config["grpo"]["weight_decay"]),
            beta=float(config["grpo"]["beta"]),
            dtype="bfloat16",
            log_wandb=config["grpo"]["log_wandb"],
            wandb_project=config["grpo"]["wandb_project"],
            # using_lora=True,
            dataset_feild="item",
            use_vllm=True,
            num_policy_updates=8,
        )

        # Initialize and train GRPO
        print("[blue]Initializing GRPO trainer[/blue]")
        grpo = GRPO(
            model=config["model_name"],
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dataset,
            reward_functions=[response_format_reward],
            config=grpo_config,
            vllm_client=vllm_client,
            
        )

        print("[blue]Starting training[/blue]")
        grpo.train()

    except Exception as e:
        print(f"[red]Training failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
