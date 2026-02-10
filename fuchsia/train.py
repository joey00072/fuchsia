#!/usr/bin/env python3
"""Generic Fuchsia trainer entrypoint.

This module centralizes the training bootstrap logic so examples only need to
define server-side rollout and reward behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

from fuchsia.config import FuchsiaConfig
from fuchsia.cpu_offloading import apply_cpu_gradient_checkpoint_monkey_patch
from fuchsia.dist_dataset import DatasetClient, PreparedRolloutBatchDataset
from fuchsia.trainer import Trainer
from fuchsia.vllm_client import VLLMClient


def _resolve_client_host(config: FuchsiaConfig) -> str:
    # 0.0.0.0 is a bind address, but the client should connect to localhost.
    if config.host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return config.host


def run_training(config_path: str | Path) -> None:
    trainer_config = FuchsiaConfig.from_yaml(str(config_path))

    vllm_client = VLLMClient(
        host=_resolve_client_host(trainer_config),
        server_port=trainer_config.port,
        init_communicator=not trainer_config.single_gpu,
    )
    if trainer_config.single_gpu:
        vllm_client.sleep()

    rollout_stream = DatasetClient(
        vllm_client,
        transfer_mode=trainer_config.sample_transfer_mode,
        queue_dir=trainer_config.sample_transfer_dir,
        poll_interval=trainer_config.sample_transfer_poll_interval,
    )

    print("CUDA AVAILABLE:", torch.cuda.is_available())

    enable_gradient_checkpointing = trainer_config.gradient_checkpointing_enabled
    cpu_offloading = trainer_config.gradient_checkpointing_cpu_offloading
    if enable_gradient_checkpointing and cpu_offloading:
        apply_cpu_gradient_checkpoint_monkey_patch()

    model_name = trainer_config.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        use_cache=False,
        torch_dtype=trainer_config.trainer_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    dataset = PreparedRolloutBatchDataset(
        source=rollout_stream,
        tokenizer=tokenizer,
        batch_size=trainer_config.batch_size,
        device=trainer_config.device,
        dtype=trainer_config.trainer_dtype,
        debug=trainer_config.debug,
        non_blocking=trainer_config.non_blocking,
    )

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
        weight_decay=trainer_config.weight_decay,
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
    trainer.train(max_iterations=trainer_config.max_iterations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Fuchsia generic trainer")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args(argv)
    run_training(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
