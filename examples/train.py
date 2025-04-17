from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from rich import print  
import math   
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from fuchsia.grpo import GRPO, GRPOConfig
from fuchsia.dist_dataset import DatasetClient
from fuchsia.vllm_client import VLLMClient


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
        # Load configuration
        config_path = Path(__file__).parent / "config.yaml"
        config = load_config(str(config_path))

        # Extract configuration variables
        # Model configuration
        model_name = config["model"]["name"]
        model_dtype = config["model"]["dtype"]
        
        # LoRA configuration
        lora_enabled = config["lora"]["enabled"]
        lora_r = config["lora"]["r"]
        lora_alpha = config["lora"]["alpha"]
        lora_target_modules = config["lora"]["target_modules"]
        
        # GRPO configuration
        grpo_group_size = config["grpo"]["group_size"]
        grpo_micro_group_size = config["grpo"]["micro_group_size"]
        grpo_batch_size = config["grpo"]["batch_size"]
        grpo_lr = float(config["grpo"]["lr"])
        grpo_weight_decay = float(config["grpo"]["weight_decay"])
        grpo_beta = float(config["grpo"]["beta"])
        grpo_epsilon = float(config["grpo"]["epsilon"])
        grpo_log_wandb = config["grpo"]["log_wandb"]
        grpo_wandb_project = config["grpo"]["wandb_project"]
        grpo_num_policy_updates = config["grpo"]["num_policy_updates"]
        grpo_lora_path = config["grpo"]["lora_path"]
        
        # Dataset configuration
        dataset_field = config["dataset"]["field"]

        # Initialize model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure LoRA
        if lora_enabled:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
            )
            model = get_peft_model(model, lora_config)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=grpo_lr, 
            weight_decay=grpo_weight_decay
        )

        # Initialize VLLM client
        vllm_client = VLLMClient()
        dataset = DatasetClient(vllm_client)

        # Configure GRPO
        grpo_config = GRPOConfig(
            group_size=grpo_group_size,
            micro_group_size=grpo_micro_group_size,
            batch_size=grpo_batch_size,
            lr=grpo_lr,
            weight_decay=grpo_weight_decay,
            beta=grpo_beta,
            epsilon=grpo_epsilon,
            dtype=model_dtype,
            log_wandb=grpo_log_wandb,
            wandb_project=grpo_wandb_project,
            dataset_feild=dataset_field,
            use_vllm=True,
            num_policy_updates=grpo_num_policy_updates,
            using_lora=isinstance(model, PeftModel),
            lora_path=grpo_lora_path,
        )

        # Initialize and train GRPO
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
