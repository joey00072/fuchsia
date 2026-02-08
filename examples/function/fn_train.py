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

from fuchsia.trainer import Trainer
from fuchsia.config import FuchsiaConfig
from fuchsia.dist_dataset import DatasetClient
from fuchsia.vllm_client import VLLMClient
from fuchsia.cpu_offloading import apply_cpu_gradient_checkpoint_monkey_patch

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    trainer_config = FuchsiaConfig.from_yaml(Path(__file__).parent / "config.yaml")
    vllm_client = VLLMClient(init_communicator=False)
    vllm_client.sleep()
    dataset = DatasetClient(
        vllm_client,
        transfer_mode=trainer_config.sample_transfer_mode,
        queue_dir=trainer_config.sample_transfer_dir,
        poll_interval=trainer_config.sample_transfer_poll_interval,
    )
    
    print("CUDA AVAILABLE:", torch.cuda.is_available())
    
    
    enable_gradient_checkpointing = trainer_config.gradient_checkpointing_enabled
    cpu_offloading = trainer_config.gradient_checkpointing_cpu_offloading
    if enable_gradient_checkpointing and cpu_offloading:
        print("[blue]Applying CPU gradient checkpoint monkey patch[/blue]")
        apply_cpu_gradient_checkpoint_monkey_patch()

    model_name = trainer_config.model_name
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
        r=trainer_config.lora_r,
        lora_alpha=trainer_config.lora_alpha,
        target_modules=trainer_config.lora_target_modules,
    )
    if trainer_config.using_lora:
        model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=trainer_config.lr,
        weight_decay=trainer_config.weight_decay
    )
    
    model.train()

    print("[blue]Initializing trainer[/blue]")
    trainer = Trainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,    
        config=trainer_config,
        vllm_client=vllm_client,
    )

    print("[blue]Starting training[/blue]")
    trainer.train()


if __name__ == "__main__":
    main()
