from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from rich import print
import math
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from fuchsia.grpo import GRPO
from fuchsia.grpo_config import GRPOConfig
from fuchsia.dist_dataset import DatasetClient
from fuchsia.vllm_client import VLLMClient
import json
import re
import regex

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    """Prepare the JEE dataset with better error handling and validation."""

    def format_prompt(row):
        prompt = "give correct answer boxed\nQuestion:" + row["question"]
        if row["Type"] == "MCQ":
            options = json.loads(row["options"])
            for option in options:
                prompt += f"\n{option['identifier']}. {option['content']}"

        message = [{"role": "user", "content": prompt}]
        row["text"] = tokenizer.apply_chat_template(message, tokenize=False)
        if "MCQ" in row["Type"]:
            option = row["correct_option"]
            answer = {"1":"A","2":"B","3":"C","4":"D"}[option]
        else:
            answer = row["correct_answer"]
        
        row["answer"] = answer
        return row

    # Apply formatting and filtering
    dataset = dataset.map(format_prompt).filter(lambda x: int(x["pass_rate"]) <= 8)
    return dataset


def find_boxes(text):
    pattern = r"boxed\{((?:[^{}]+|(?R))*)\}"
    matches = regex.finditer(pattern, text)
    boxes = []
    for match in matches:
        boxes.append(match.group(1))
    return boxes


def response_format_reward(sample: dict, s: str, *args, **kwargs) -> float:
    """Reward function for JEE dataset."""
    boxes = find_boxes(s)
    if not boxes:
        return 0.0
    
    last_box = boxes[-1]
    if last_box.strip() == sample["answer"]:
        return 1.0
    return 0.0


import requests


class MockClient:
    def __init__(self):
        self.buffer = []
    
    def get_sample(self):
        if not self.buffer:
            return None
        return self.buffer.pop(0)
    
    def update_model_params(self, model):
        pass
    
    def empty_buffer(self):
        self.buffer = []
    
    def fill_buffer(self):
        pass


def main():
    global tokenizer

    # Load configuration
    config_path = Path(__file__).parent / "ds_config.yaml"
    config = load_config(str(config_path))

    # Initialize model and tokenizer
    print("[blue]Loading model and tokenizer...[/blue]")
    model_name = config["model"]["name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("[green]Model and tokenizer loaded successfully[/green]")
    
    # Configure LoRA if enabled BEFORE optimizer
    if config["lora"]["enabled"]:
        print("[blue]Configuring LoRA...[/blue]")
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        print("[green]LoRA configured successfully[/green]")

    # Now create optimizer after LoRA is applied
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["grpo"]["lr"]),
        weight_decay=float(config["grpo"]["weight_decay"])
    )

    # print model
    print("[blue]Model:[/blue]")
    print(model)

    # Initialize VLLM client
    print("[blue]Initializing VLLM client...[/blue]")
    vllm_client = VLLMClient()
    
    # Initialize dataset client
    print("[blue]Initializing dataset client...[/blue]")
    dataset_client = DatasetClient(vllm_client)

    # Configure GRPO
    print("[blue]Configuring GRPO...[/blue]")
    grpo_config = GRPOConfig(
        group_size=config["grpo"]["group_size"],
        micro_group_size=config["grpo"]["micro_group_size"],
        batch_size=config["grpo"]["batch_size"],
        lr=float(config["grpo"]["lr"]),
        weight_decay=float(config["grpo"]["weight_decay"]),
        beta=float(config["grpo"]["beta"]),
        epsilon=float(config["grpo"]["epsilon"]),
        log_wandb=config["grpo"]["log_wandb"],
        wandb_project=config["grpo"]["wandb_project"],
        num_policy_updates=config["grpo"]["num_policy_updates"],
        using_lora=config["lora"]["enabled"],
        lora_path=config["grpo"]["lora_path"],
        use_vllm=True,
        dataset_feild="text",
        ignore_imcomplete_samples=True
    )

    # Initialize and train GRPO
    print("[blue]Initializing GRPO trainer...[/blue]")
    grpo = GRPO(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset_client,
        optimizer=optimizer,
        reward_functions=[response_format_reward],
        config=grpo_config,
        vllm_client=vllm_client
    )

    print("[blue]Starting training...[/blue]")
    grpo.train()



if __name__ == "__main__":
    main()
