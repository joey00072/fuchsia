import os
import math
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml
from rich import print
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from fuchsia.grpo import GRPO
from fuchsia.grpo_config import GRPOConfig
from fuchsia.dist_dataset import DatasetClient
from fuchsia.vllm_client import VLLMClient
# from fuchsia.cpu_offloding import apply_unsloth_offloaded_gradient_checkpoint_monkey_patch

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()

SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"


def load_config(config_path: str) -> Dict[str, Any]:
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
    
    END_OF_TEXT_TOKEN     =  "<|eot_id|>"
    START_HEADER_TOKEN    =  "<|start_header_id|>"
    END_HEADER_TOKEN      =  "<|end_header_id|>"
    ASSISTANT_TOKEN       =  "assistant"
    USER_TOKEN            =  "user"

    START_THINKING_TOKEN  =  "<thinking>"
    END_THINKING_TOKEN    =  "</thinking>"
    START_ANSWER_TOKEN    =  "<answer>"
    END_ANSWER_TOKEN      =  "</answer>"

    try:
        try:
            s = s.split(
                f"{END_OF_TEXT_TOKEN}{START_HEADER_TOKEN}{ASSISTANT_TOKEN}{END_HEADER_TOKEN}"
            )[1]
        except IndexError:
            return -1.0

        if END_OF_TEXT_TOKEN in s:
            s = s.split(END_OF_TEXT_TOKEN)[0]

        format_reward = 0.0
        content_reward = 0.0
        correct_template = 0

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

        if s.count("<thinking>") == 1:
            format_reward += 0.5
            thinking_content = (
                s.split(START_THINKING_TOKEN)[1].split(END_THINKING_TOKEN)[0].strip()
            )
            if len(thinking_content) > 10:  # Basic content validation
                content_reward += 0.5
        else:
            format_reward -= 0.1

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
        response = requests.post(url)
        return response.json()["sample"]

    def update_model_params(self, model):
        pass

    def empty_buffer(self):
        pass

    def fill_buffer(self):
        pass


def main():
    try:
        
        print("CUDA AVAILABLE:", torch.cuda.is_available())
        config_path = Path(__file__).parent / "gsm8k_config.yaml"
        config = load_config(str(config_path))

        model_name = config["model"]["name"]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"].get("target_modules", None),
        )
        if config["lora"].get("enabled", True):
            model = get_peft_model(model, lora_config)
            model.gradient_checkpointing_enable()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config["grpo"]["lr"]),
            weight_decay=float(config["grpo"]["weight_decay"])
        )

        model.gradient_checkpointing_enable()
        model._prepare_model_for_gradient_checkpointing(model)
        model.enable_input_require_grads()

        print("[blue]Loading GSM8K dataset[/blue]")
        dataset = load_dataset("openai/gsm8k", "main")["train"]
        dataset = prepare_dataset(dataset)
        vllm_client = VLLMClient(init_communicator=False)
        dataset = DatasetClient(vllm_client)

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
            dataset_feild="item",
            use_vllm=True,
            num_policy_updates=config["grpo"]["num_policy_updates"],
            single_gpu=config["grpo"]["single_gpu"],
        )

        print("[blue]Initializing GRPO trainer[/blue]")
        grpo = GRPO(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dataset,
            optimizer=optimizer,
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
