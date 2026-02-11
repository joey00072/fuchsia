# derived from https://github.com/unslothai/

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this is referred to unsloth's cpu offloading implementation
# but its different and intentionally lightweight.

from __future__ import annotations

from collections.abc import Sequence

import torch
import transformers

_CHECKPOINT_WRAPPER_KWARGS = {
    "preserve_rng_state",
    "use_reentrant",
    "determinism_check",
    "debug",
    "context_fn",
    "early_stop",
}


class CPUGradientCheckpointer(torch.autograd.Function):
    """Checkpoint the first activation tensor by parking it on CPU between fwd/bwd."""

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
    """
    Checkpoint function compatible with checkpoint_wrapper(checkpoint_fn=...).
    It offloads the first floating activation tensor to CPU between fwd and bwd.
    """
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

    return CPUGradientCheckpointer.apply(run_fn, hidden_states)


def apply_cpu_gradient_checkpoint_monkey_patch():
    def new_gradient_checkpointing_enable(self):
        self._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=cpu_offloaded_checkpoint,
        )
        if getattr(self, "_hf_peft_config_loaded", False):
            self.enable_input_require_grads()

    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = (
        new_gradient_checkpointing_enable
    )
