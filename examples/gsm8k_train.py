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
from fuchsia.cpu_offloding import apply_cpu_gradient_checkpoint_monkey_patch

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
apply_cpu_gradient_checkpoint_monkey_patch()

SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[red]Failed to load config from {config_path}: {e}[/red]")
        raise


class MockClient:
    def __init__(self):
        import json
        with open("mock_data.jsonl", "r") as f:
            self.samples = [json.loads(line) for line in f]

    def get_sample(self):
        return self.samples.pop(0)
    def update_model_params(self, model):
        pass

    def empty_buffer(self):
        pass

    def fill_buffer(self):
        pass
    
    def sleep(*args, **kwargs):
        pass


def main():
    try:
        vllm_client = VLLMClient(init_communicator=False)
        # vllm_client = MockClient()
        vllm_client.sleep()
        dataset = DatasetClient(vllm_client)
        
        print("CUDA AVAILABLE:", torch.cuda.is_available())
        config_path = Path(__file__).parent / "gsm8k_config.yaml"
        config = load_config(str(config_path))

        model_name = config["model"]["name"]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            # load_in_4bit=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"].get("target_modules", None),
        )
        if config["lora"].get("enabled"):
            model = get_peft_model(model, lora_config)
            model.gradient_checkpointing_enable()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config["grpo"]["lr"]),
            weight_decay=float(config["grpo"]["weight_decay"])
        )

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.train()



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
            using_lora=config["lora"].get("enabled", False),
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
            reward_functions=[],
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
