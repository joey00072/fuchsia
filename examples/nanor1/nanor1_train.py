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

def main():
    grpo_config = GRPOConfig.from_yaml(Path(__file__).parent / "nanor1_config.yaml")
    vllm_client = VLLMClient(init_communicator=not grpo_config.single_gpu)
    if grpo_config.single_gpu:
        vllm_client.sleep()
    dataset = DatasetClient(
        vllm_client,
        transfer_mode=grpo_config.sample_transfer_mode,
        queue_dir=grpo_config.sample_transfer_dir,
        poll_interval=grpo_config.sample_transfer_poll_interval,
    )
    
    print("CUDA AVAILABLE:", torch.cuda.is_available())
    
    
    enable_gradient_checkpointing = grpo_config.gradient_checkpointing_enabled
    cpu_offloading = grpo_config.gradient_checkpointing_cpu_offloading
    if enable_gradient_checkpointing and cpu_offloading:
        apply_cpu_gradient_checkpoint_monkey_patch()

    model_name = grpo_config.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        # load_in_4bit=True,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        r=grpo_config.lora_r,
        lora_alpha=grpo_config.lora_alpha,
        target_modules=grpo_config.lora_target_modules,
    )
    if grpo_config.using_lora:
        model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=grpo_config.lr,
        weight_decay=grpo_config.weight_decay
    )
    
    model.train()

    print("[blue]Initializing GRPO trainer[/blue]")
    grpo = GRPO(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,    
        config=grpo_config,
        vllm_client=vllm_client,
    )

    print("[blue]Starting training[/blue]")
    grpo.train()


if __name__ == "__main__":
    main()
