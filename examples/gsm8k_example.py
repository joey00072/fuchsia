from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
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
        with open(config_path, 'r') as f:
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
            s = s.split(f"{END_OF_TEXT_TOKEN}{START_HEADER_TOKEN}{ASSISTANT_TOKEN}{END_HEADER_TOKEN}")[1]
        except IndexError:
            return -1.0

        if END_OF_TEXT_TOKEN in s:
            s = s.split(END_OF_TEXT_TOKEN)[0]

        # Initialize reward components
        format_reward = 0.0
        content_reward = 0.0
        correct_template = 0

        # Check format tags
        required_tags = [START_THINKING_TOKEN, END_THINKING_TOKEN, START_ANSWER_TOKEN, END_ANSWER_TOKEN]
        for tag in required_tags:
            if tag in s:
                format_reward += 0.15
                if s.count(tag) > 1:
                    format_reward -= s.count(tag) * 0.01

        # Validate thinking section
        if s.count("<thinking>") == 1:
            format_reward += 0.5
            thinking_content = s.split(START_THINKING_TOKEN)[1].split(END_THINKING_TOKEN)[0].strip()
            if len(thinking_content) > 10:  # Basic content validation
                content_reward += 0.5
        else:
            format_reward -= 0.1

        # Validate answer section
        if "<answer>" in s and "</answer>" in s:
            format_reward += 0.4
            answer_content = s.split(START_ANSWER_TOKEN)[1].split(END_ANSWER_TOKEN)[0].strip()
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

def main():
    try:
        # Load configuration
        config_path = Path(__file__).parent / "gsm8k_config.yaml"
        config = load_config(str(config_path))
        
        # Initialize model and tokenizer
        print(f"[blue]Loading model: {config['model_name']}[/blue]")
        model = AutoModelForCausalLM.from_pretrained(config['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Configure LoRA
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            target_modules=config['lora'].get('target_modules', None)
        )
        model = get_peft_model(model, lora_config)
        model = model.to(torch.bfloat16)

        # Load and prepare dataset
        print("[blue]Loading GSM8K dataset[/blue]")
        # dataset = load_dataset("openai/gsm8k", "main")["train"]
        # dataset = prepare_dataset(dataset)
        vllm_client = VLLMClient()
        dataset = DatasetClient(vllm_client)

        # Configure GRPO
        grpo_config = GRPOConfig(
            group_size=config['grpo']['group_size'],
            micro_group_size=config['grpo']['micro_group_size'],
            batch_size=config['grpo']['batch_size'],
            lr=float(config['grpo']['lr']),
            weight_decay=float(config['grpo']['weight_decay']),
            beta=float(config['grpo']['beta']),
            dtype="bfloat16", 
            log_wandb=config['grpo']['log_wandb'],
            wandb_project=config['grpo']['wandb_project'],
            # using_lora=True, 
            dataset_feild="item",
            use_vllm=True,
            num_policy_updates=8,
        )

        # Initialize and train GRPO
        print("[blue]Initializing GRPO trainer[/blue]")
        grpo = GRPO(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dataset,
            reward_functions=[response_format_reward],
            config=grpo_config,
            vllm_client=vllm_client
        )

        print("[blue]Starting training[/blue]")
        grpo.train()

    except Exception as e:
        print(f"[red]Training failed: {e}[/red]")
        raise

if __name__ == "__main__":
    main() 