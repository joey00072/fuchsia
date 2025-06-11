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

# this is refered to unsloth's cpu offloading implementation
# but its different and not as good as unsloth's :)

import torch
import transformers
import inspect


class CPUGradientCheckpointer(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_fn, activations, *kwargs):
        
        with torch.no_grad():
            output = forward_fn(activations, *kwargs)
            
        cpu_activations = activations.to("cpu", non_blocking=True)
        
        ctx.save_for_backward(cpu_activations)
        ctx.forward_fn = forward_fn
        ctx.kwargs = kwargs

        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        
        (cpu_activations,) = ctx.saved_tensors
        
        gpu_activations = cpu_activations.to("cuda", non_blocking=True).detach()
        gpu_activations.requires_grad = True
        
        with torch.enable_grad():
            (output,) = ctx.forward_fn(gpu_activations, *ctx.kwargs)
            
        torch.autograd.backward(output, grad_output)
        
        return (None, gpu_activations.grad,) + (None,) * len(ctx.kwargs)




def apply_cpu_gradient_checkpoint_monkey_patch():
    def new_gradient_checkpointing_enable(self):
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=CPUGradientCheckpointer.apply)

        if getattr(self, "_hf_peft_config_loaded", False):
            self.enable_input_require_grads()
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = new_gradient_checkpointing_enable