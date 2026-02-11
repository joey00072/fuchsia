from __future__ import annotations

from typing import Optional

import torch.nn as nn

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
except Exception:  # pragma: no cover - depends on torch build
    checkpoint_wrapper = None

from fuchsia.cpu_offloading import cpu_offloaded_checkpoint

CHECKPOINT_MODES = {"activation", "activation_offload"}


def _resolve_transformer_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    # Unwrap PEFT model first when present.
    root = model
    if hasattr(root, "base_model") and hasattr(root.base_model, "model"):
        root = root.base_model.model

    candidates = (
        ("model", "layers"),   # Llama/Qwen/Mistral HF causal LM
        ("layers",),           # direct language model
        ("transformer", "h"),  # GPT-style
    )
    for path in candidates:
        node = root
        try:
            for attr in path:
                node = getattr(node, attr)
        except AttributeError:
            continue
        if isinstance(node, nn.ModuleList):
            return node
    return None


def apply_activation_checkpointing(
    model: nn.Module,
    *,
    freq: int = 1,
    preserve_rng_state: bool = False,
    mode: str = "activation",
) -> int:
    """
    Apply activation checkpointing to transformer blocks by wrapping layers in-place.
    Returns the number of wrapped layers.
    """
    if checkpoint_wrapper is None:
        return 0
    if freq <= 0:
        raise ValueError(f"freq must be > 0, got {freq}")
    if mode not in CHECKPOINT_MODES:
        raise ValueError(
            f"Unsupported activation checkpoint mode: {mode}. "
            f"Supported values: {sorted(CHECKPOINT_MODES)}"
        )

    layers = _resolve_transformer_layers(model)
    if layers is None:
        return 0

    checkpoint_kwargs = {"preserve_rng_state": preserve_rng_state}
    if mode == "activation_offload":
        checkpoint_kwargs["checkpoint_fn"] = cpu_offloaded_checkpoint

    wrapped = 0
    for idx, layer in enumerate(layers):
        if idx % freq != 0:
            continue
        # Avoid double-wrapping if called more than once.
        if hasattr(layer, "_checkpoint_wrapped_module"):
            continue
        layers[idx] = checkpoint_wrapper(layer, **checkpoint_kwargs)
        wrapped += 1
    return wrapped
