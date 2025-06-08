from dataclasses import dataclass, field
from vllm import LLM, SamplingParams
from typing import Callable
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from rich import print

from copy import deepcopy

@dataclass
class Rollout:
    
    prompt: str = ""
    completion: str = ""
    last_completion: str = ""
    stop: list[str] = field(default_factory=list)
    stop_reason: str = ""
    finish_reason: str = ""
    completed: bool = False
    state: dict = field(default_factory=dict)
    completion_ids: list[int] = field(default_factory=list)
    
    
    item: dict = field(default_factory=dict)
    epoch: int = 0
    
    rewards: dict = field(default_factory=dict)
    all_rewards: dict = field(default_factory=dict)
    mean: float = 0.0
    std: float = 0.0

    def __post_init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def is_completed(self):
        return self.finish_reason in ["stop", "length"]

    @property
    def input(self):
        return self.prompt + self.completion
    
    
    def clone(self):
        return deepcopy(self)


@dataclass
class Environment:
    reward_functions: list[Callable] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    max_samples: int = 1
    
    def generate(
        self,   
        rollouts: list[Rollout] | Rollout,
        llm: LLM,
        sampling_params: SamplingParams | None = None,
        vllm_generate_kwargs: dict = {},
        **kwargs,
    ):
        
        # Environment sampling params are prioritized over the ones passed in the generate call
        if self.sampling_params is not None:
            sampling_params = self.sampling_params
            
        sampling_params = deepcopy(sampling_params)
        

        ##  You can't n>1 for multi-turn environments
        ##  otherwise the rollouts will forked at each step
        
        rollouts = [rollouts] if isinstance(rollouts, Rollout) else rollouts
        all_rollouts = []
        for rollout in rollouts:
            all_rollouts.extend([rollout.clone() for _ in range(sampling_params.n)])
            
        sampling_params.n = 1

        step = 0
        while (
            step < self.max_samples and
            not all([rollout.completed for rollout in all_rollouts])
            ):
            
            step += 1
            inputs = [rollout.input for rollout in all_rollouts]
            
            vllm_outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                **vllm_generate_kwargs,
            )
            
            ##  Update the rollouts with the vllm outputs
            for rollout, output in zip(all_rollouts, vllm_outputs):
                rollout.last_completion = output.outputs[0].text
                rollout.completion += output.outputs[0].text
                rollout.completion_ids.extend(list(output.outputs[0].token_ids))
                rollout.stop_reason = output.outputs[0].stop_reason if output.outputs[0].stop_reason else ""
                rollout.finish_reason = output.outputs[0].finish_reason if output.outputs[0].finish_reason else ""
                
            ##  Process the rollouts
            all_rollouts = self.process_rollouts(all_rollouts,step)
        return all_rollouts
    
    def process_rollouts(self, rollouts: list[Rollout], step: int):
        for rollout in rollouts:
            if rollout.finish_reason in ["stop", "length"]:
                rollout.completed = True
        return rollouts
    
    def payload(self, rollouts: list[Rollout] | Rollout, calculate_rewards: bool = True):
        rollouts = [rollouts] if isinstance(rollouts, Rollout) else rollouts
        
        # Group rollouts by their original prompt/item (since we clone rollouts for n>1)
        grouped_rollouts = {}
        for rollout in rollouts:
            key = rollout.prompt  # Use prompt as the grouping key
            if key not in grouped_rollouts:
                grouped_rollouts[key] = []
            grouped_rollouts[key].append(rollout)
        
        samples = []
        for prompt, group in grouped_rollouts.items():
            # Take the first rollout to get shared properties
            first_rollout = group[0]
            
            output = {
                "item": [first_rollout.item] * len(group),
                "completions": [],
                "completion_ids": [],
                "stop_reason": [],
                "finish_reason": [],
                "epoch": first_rollout.epoch,
                "inputs": first_rollout.prompt
            }
            
            # Add data from each rollout in the group
            for rollout in group:
                output["completions"].append(rollout.completion)
                output["completion_ids"].append(rollout.completion_ids)
                output["stop_reason"].append(rollout.stop_reason)
                output["finish_reason"].append(rollout.finish_reason)

            if calculate_rewards and self.reward_functions:
                output["all_rewards"], output["rewards"], output["mean"], output["std"] = (
                    self.calculate_rewards(output["item"], output["completions"], output["completion_ids"])
                )
            else:
                                 output["all_rewards"], output["rewards"], output["mean"], output["std"] = {}, [], 0.0, 0.0
                 
            samples.append(output)
        samples = [s for s in samples if s["std"] != 0.0]
        return samples

    def calculate_rewards(self, items, completions, completion_ids):
        """Calculate rewards using the environment's reward functions."""
        import numpy as np
        
        all_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function(
                None, items, completions, completion_ids  # tokenizer=None for now
            )
            all_rewards[reward_function.__name__] = rewards

        # Convert all reward lists to numpy arrays and stack them
        if all_rewards:
            reward_arrays = []
            for rewards in all_rewards.values():
                reward_array = np.array(rewards)
                reward_arrays.append(reward_array)
            
            # Stack arrays if we have multiple reward functions, otherwise use the single array
            if len(reward_arrays) > 1:
                reward_values = np.stack(reward_arrays, axis=0)
                total_rewards = reward_values.sum(axis=0)
            else:
                total_rewards = reward_arrays[0]
            
            mean = float(np.mean(total_rewards))
            std = float(np.std(total_rewards))
            
            return all_rewards, total_rewards.tolist(), mean, std
        else:
            return {}, [], 0.0, 0.0
            
    

    
@dataclass
class SingleTurnEnvironment(Environment):
    def process_rollouts(self, rollouts: list[Rollout], step: int):
        for rollout in rollouts:
            if rollout.finish_reason in ["stop", "length"]:
                rollout.completed = True
        return rollouts


class MultiTurnEnvironment(Environment):
    def process_rollouts(self, rollouts: list[Rollout], step: int):
        for rollout in rollouts:
            if rollout.finish_reason in ["stop", "length"]:
                rollout.completed = True
        for rollout in rollouts:
            MultiTurnEnvironment.step_rollout(rollout)
        return rollouts

    @staticmethod
    def step_rollout(rollout: Rollout):
        assert False, "Not implemented"


class PythonEnvironment(MultiTurnEnvironment):
    def __init__(self):
        self.stop = ["</python>"]

    @staticmethod
    def python(code: str) -> str:
        import subprocess

        try:
            result = subprocess.run(
                ["python", "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True,
            )
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            output = result.stdout.strip() if result.stdout else ""
            if len(output) > 512:
                output = output[:512] + "... (truncated to 512 chars)"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 10 seconds"

    @staticmethod
    def step_rollout(rollout: Rollout):
        if rollout.stop_reason != "</python>":
            return rollout

        if not "<python>" in rollout.last_completion:
            return rollout

        code = rollout.last_completion.split("<python>")[1]

        output = PythonEnvironment.python(code)

        rollout.last_completion = f"</python>\n<output>\n{output}\n</output>"
        return rollout

