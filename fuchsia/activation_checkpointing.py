from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
except Exception:  # pragma: no cover - depends on torch build
    checkpoint_wrapper = None

CHECKPOINT_MODES = {"activation", "activation_offload"}
_CHECKPOINT_WRAPPER_KWARGS = {
    "preserve_rng_state",
    "use_reentrant",
    "determinism_check",
    "debug",
    "context_fn",
    "early_stop",
}


class _CPUOffloadedCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_fn, activations):
        with torch.no_grad():
            outputs = run_fn(activations)

        ctx.run_fn = run_fn
        ctx.device = activations.device
        ctx.save_for_backward(activations.to("cpu", non_blocking=True))
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        (cpu_activations,) = ctx.saved_tensors
        replay_activations = cpu_activations.to(ctx.device, non_blocking=True).detach()
        replay_activations.requires_grad_(True)

        with torch.enable_grad():
            outputs = ctx.run_fn(replay_activations)

        if torch.is_tensor(outputs):
            outputs_seq = (outputs,)
        elif isinstance(outputs, Sequence):
            outputs_seq = tuple(outputs)
        else:
            raise TypeError(
                f"Unsupported checkpoint output type: {type(outputs)!r}"
            )

        tensor_outputs: list[torch.Tensor] = []
        tensor_grads: list[torch.Tensor] = []
        for idx, out in enumerate(outputs_seq):
            if not torch.is_tensor(out) or not out.requires_grad:
                continue
            grad = grad_outputs[idx] if idx < len(grad_outputs) else None
            if grad is None:
                if out.numel() != 1:
                    raise RuntimeError(
                        "Missing gradient for non-scalar checkpoint output"
                    )
                grad = torch.ones_like(out)
            tensor_outputs.append(out)
            tensor_grads.append(grad)

        if tensor_outputs:
            torch.autograd.backward(tensor_outputs, tensor_grads)

        return None, replay_activations.grad


def _find_primary_activation_index(args: tuple[object, ...]) -> int | None:
    for idx, value in enumerate(args):
        if torch.is_tensor(value) and value.is_floating_point():
            return idx
    return None


def cpu_offloaded_checkpoint(function, *args, **kwargs):
    if not args:
        return function(*args, **kwargs)

    forward_kwargs = dict(kwargs)
    for key in _CHECKPOINT_WRAPPER_KWARGS:
        forward_kwargs.pop(key, None)

    hidden_idx = _find_primary_activation_index(args)
    if hidden_idx is None:
        return function(*args, **forward_kwargs)

    hidden_states = args[hidden_idx]
    prefix = args[:hidden_idx]
    suffix = args[hidden_idx + 1 :]

    def run_fn(replay_hidden_states):
        call_args = (*prefix, replay_hidden_states, *suffix)
        return function(*call_args, **forward_kwargs)

    return _CPUOffloadedCheckpoint.apply(run_fn, hidden_states)


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
