from dataclasses import dataclass, field
from vllm import LLM, SamplingParams
from typing import Callable
import numpy as np
import uuid
from concurrent.futures import ThreadPoolExecutor
from rich import print
from collections import defaultdict
from copy import deepcopy
from transformers import AutoTokenizer


@dataclass
class Rollout:
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
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
        new_rollout = deepcopy(self)
        new_rollout.id = str(uuid.uuid4())
        new_rollout.group_id = self.group_id
        return new_rollout


@dataclass
class Environment:
    reward_functions: list[Callable] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    max_steps: int = 1
    stop: list[str] = field(default_factory=list)
    
    def generate(
        self,   
        rollouts: list[Rollout] | Rollout,
        llm: LLM,
        sampling_params: SamplingParams | None = None,
        vllm_generate_kwargs: dict = {},
        tokenizer: AutoTokenizer = None,
        **kwargs,
    ):
        
        print(self)
        print("--------------------------------")
        process_kwargs = {
            "tokenizer": tokenizer,
            "max_model_len": llm.llm_engine.model_config.max_model_len,
        }
        # Environment sampling params are prioritized over the ones passed in the generate call
        if self.sampling_params is not None:
            sampling_params = self.sampling_params
            
        sampling_params:SamplingParams = deepcopy(sampling_params)
        
        if self.stop:
            sampling_params.stop = self.stop
        ##  You can't n>1 for multi-turn environments
        ##  otherwise the rollouts will forked at each step
        
        rollouts = [rollouts] if isinstance(rollouts, Rollout) else rollouts
        all_rollouts: dict[str, Rollout] = {rollout.id: rollout for rollout in rollouts}
        
        
        step = 0
        while (
            step < self.max_steps and
            not all([rollout.completed for rollout in all_rollouts.values()])
            ):
            
            step += 1
            active_rollouts = [rollout for rollout in all_rollouts.values() if not rollout.completed]
            inputs = [rollout.input for rollout in active_rollouts]
            
            vllm_outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                **vllm_generate_kwargs,
            )
            
            ##  Update the rollouts with the vllm outputs
            new_rollouts = []
            for rollout, outputs in zip(active_rollouts, vllm_outputs):
                for idx,output in enumerate(outputs.outputs):
                    new_rollout = rollout.clone() if idx != len(outputs.outputs) - 1 else rollout
                    new_rollout.last_completion = output.text
                    new_rollout.completion += output.text
                    new_rollout.completion_ids.extend(list(output.token_ids))
                    new_rollout.stop_reason = output.stop_reason if output.stop_reason else ""
                    new_rollout.finish_reason = output.finish_reason if output.finish_reason else ""
                    new_rollouts.append(new_rollout)
            ##  Process the rollouts
            new_rollouts = self.process_rollouts(new_rollouts,step, process_kwargs)
            for rollout in new_rollouts:
                all_rollouts[rollout.id] = rollout
            
            # reset the n to 1 otherwise the rollouts will forked at each step
            sampling_params.n = 1
        
        grouped_rollouts = defaultdict(list)
        for rollout in all_rollouts.values():
            grouped_rollouts[rollout.group_id].append(rollout)
        
        output_rollouts = []
        for group in grouped_rollouts.values():
            for rollout in group:
                output_rollouts.append(rollout)
        return output_rollouts
    
    def process_rollouts(self, rollouts: list[Rollout], step: int, process_kwargs: dict):
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
                    self.calculate_rewards(output["item"], output["completions"], output["completion_ids"], group)
                )
            else:
                output["all_rewards"], output["rewards"], output["mean"], output["std"] = {}, [], 0.0, 0.0
                 
            samples.append(output)
        print(f"{[s['all_rewards'] for s in samples]}")
        samples = [s for s in samples if s["std"] != 0.0]
        return samples

    def calculate_rewards(self, items, completions, completion_ids, rollouts):
        """Calculate rewards using the environment's reward functions."""
        import numpy as np
        
        all_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function(
                rollouts=rollouts, items=items, completions=completions, completion_ids=completion_ids
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
    def process_rollouts(self, rollouts: list[Rollout], step: int, process_kwargs: dict):
        for rollout in rollouts:
            if rollout.finish_reason in ["stop", "length"]:
                rollout.completed = True
        return rollouts

@dataclass
class MultiTurnEnvironment(Environment):
    def process_rollouts(self, rollouts: list[Rollout], step: int, process_kwargs: dict):
        for rollout in rollouts:
            if (rollout.finish_reason in ["length"] 
                or (rollout.finish_reason in ["stop"] and rollout.stop_reason not in self.stop)
                or rollout.completed):
                rollout.completed = True
        for rollout in rollouts:
            self.step_rollout(rollout)
            
        tokenizer = process_kwargs["tokenizer"]
        max_model_len = process_kwargs["max_model_len"]
        for rollout in rollouts:
            token_length = len(tokenizer.encode(rollout.input))
            if token_length > max_model_len:
                rollout.completed = True
                rollout.finish_reason = "length"
                rollout.stop_reason = "length"
                rollout.stop = [tokenizer.eos_token]
                rollout.completion = tokenizer.decode(rollout.completion_ids[:max_model_len])
                
        return rollouts 

    def step_rollout(self, rollout: Rollout):
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

