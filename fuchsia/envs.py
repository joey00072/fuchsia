from dataclasses import asdict, dataclass, field
from vllm import LLM, SamplingParams
from typing import Callable, Any
import inspect
import numpy as np
import uuid
from collections import defaultdict
from copy import deepcopy
from transformers import AutoTokenizer

from fuchsia.reward_utils import clean_completions


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
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    
    
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
class RolloutSample:
    item: list[dict]
    prompt_ids: list[int]
    completions: list[str]
    completion_ids: list[list[int]]
    completion_logprobs: list[list[float]]
    stop_reason: list[str]
    finish_reason: list[str]
    epoch: int
    inputs: str
    all_rewards: dict[str, list[float]] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Environment:
    reward_functions: list[Callable] = field(default_factory=list)
    sampling_params: SamplingParams | None = None
    max_steps: int = 1
    n: int | None = None
    stop: list[str] = field(default_factory=list)

    @staticmethod
    def _coerce_logprob(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, dict):
            logprob = value.get("logprob")
            return float(logprob) if logprob is not None else 0.0
        logprob = getattr(value, "logprob", None)
        return float(logprob) if logprob is not None else 0.0

    @classmethod
    def _extract_step_logprob(cls, token_id: int, step_logprobs: Any) -> float:
        if step_logprobs is None:
            return 0.0

        if isinstance(step_logprobs, dict):
            token_data = step_logprobs.get(token_id)
            if token_data is None:
                token_data = step_logprobs.get(str(token_id))
            if token_data is None and step_logprobs:
                token_data = next(iter(step_logprobs.values()))
            return cls._coerce_logprob(token_data)

        if isinstance(step_logprobs, (list, tuple)):
            if len(step_logprobs) == 0:
                return 0.0
            for candidate in step_logprobs:
                if isinstance(candidate, dict):
                    candidate_token_id = candidate.get("token_id")
                    if candidate_token_id == token_id or str(candidate_token_id) == str(token_id):
                        return cls._coerce_logprob(candidate)
                else:
                    candidate_token_id = getattr(candidate, "token_id", None)
                    if candidate_token_id == token_id:
                        return cls._coerce_logprob(candidate)
            return cls._coerce_logprob(step_logprobs[0])

        return cls._coerce_logprob(step_logprobs)

    @classmethod
    def _extract_completion_logprobs(cls, output: Any) -> list[float]:
        token_ids = list(getattr(output, "token_ids", []))
        raw_logprobs = getattr(output, "logprobs", None)
        if not token_ids:
            return []
        if raw_logprobs is None:
            return [0.0] * len(token_ids)

        completion_logprobs: list[float] = []
        for idx, token_id in enumerate(token_ids):
            step_logprobs = raw_logprobs[idx] if idx < len(raw_logprobs) else None
            completion_logprobs.append(cls._extract_step_logprob(token_id, step_logprobs))
        return completion_logprobs

    def _mark_length_limited_rollout(
        self,
        rollout: Rollout,
        *,
        tokenizer: AutoTokenizer | None,
        max_model_len: int,
        token_length: int,
    ) -> None:
        rollout.completed = True
        rollout.finish_reason = "length"
        rollout.stop_reason = "length"
        if tokenizer is not None and tokenizer.eos_token is not None:
            rollout.stop = [tokenizer.eos_token]
        rollout.state["context_token_length"] = int(token_length)
        rollout.state["context_limit"] = int(max_model_len)
        rollout.state["context_limit_exceeded"] = True

        # Keep token arrays consistent if completion ids exceed remaining context.
        if rollout.prompt_ids:
            max_completion_len = max(0, max_model_len - len(rollout.prompt_ids))
            if len(rollout.completion_ids) > max_completion_len:
                rollout.completion_ids = rollout.completion_ids[:max_completion_len]
                rollout.completion_logprobs = rollout.completion_logprobs[:max_completion_len]
                if tokenizer is not None:
                    rollout.completion = tokenizer.decode(
                        rollout.completion_ids,
                        skip_special_tokens=False,
                    )

    def _enforce_context_limit(
        self,
        rollouts: list[Rollout],
        process_kwargs: dict[str, Any],
    ) -> None:
        tokenizer = process_kwargs.get("tokenizer")
        max_model_len = process_kwargs.get("max_model_len")
        if tokenizer is None or not isinstance(max_model_len, int) or max_model_len <= 0:
            return

        for rollout in rollouts:
            if rollout.completed:
                continue
            token_length = len(tokenizer.encode(rollout.input))
            if token_length >= max_model_len:
                self._mark_length_limited_rollout(
                    rollout,
                    tokenizer=tokenizer,
                    max_model_len=max_model_len,
                    token_length=token_length,
                )
    
    def generate(
        self,   
        rollouts: list[Rollout] | Rollout,
        llm: LLM,
        sampling_params: SamplingParams | None = None,
        vllm_generate_kwargs: dict | None = None,
        tokenizer: AutoTokenizer = None,
        **kwargs,
    ):
        process_kwargs = {
            "tokenizer": tokenizer,
            "max_model_len": llm.llm_engine.model_config.max_model_len,
        }
        # Environment sampling params are prioritized over the ones passed in the generate call
        if self.sampling_params is not None:
            sampling_params = self.sampling_params
            
        sampling_params:SamplingParams = deepcopy(sampling_params)
        generate_kwargs = vllm_generate_kwargs or {}
        
        if self.stop:
            sampling_params.stop = self.stop
        if self.n is not None:
            sampling_params.n = self.n
        ##  You can't n>1 for multi-turn environments
        ##  otherwise the rollouts will forked at each step
        
        rollouts = [rollouts] if isinstance(rollouts, Rollout) else rollouts
        all_rollouts: dict[str, Rollout] = {rollout.id: rollout for rollout in rollouts}
        
        
        step = 0
        while step < self.max_steps:
            self._enforce_context_limit(list(all_rollouts.values()), process_kwargs)
            if all(rollout.completed for rollout in all_rollouts.values()):
                break
            
            step += 1
            active_rollouts = [rollout for rollout in all_rollouts.values() if not rollout.completed]
            inputs = [rollout.input for rollout in active_rollouts]
            
            vllm_outputs = llm.generate(
                inputs,
                sampling_params=sampling_params,
                **generate_kwargs,
            )
            
            ##  Update the rollouts with the vllm outputs
            new_rollouts = []
            for rollout, outputs in zip(active_rollouts, vllm_outputs):
                prompt_token_ids = list(getattr(outputs, "prompt_token_ids", []) or [])
                for idx,output in enumerate(outputs.outputs):
                    new_rollout = rollout.clone() if idx != len(outputs.outputs) - 1 else rollout
                    if not new_rollout.prompt_ids and prompt_token_ids:
                        new_rollout.prompt_ids = prompt_token_ids
                    new_rollout.last_completion = output.text
                    new_rollout.completion += output.text
                    new_rollout.completion_ids.extend(list(output.token_ids))
                    new_rollout.completion_logprobs.extend(self._extract_completion_logprobs(output))
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

    def _group_rollouts(self, rollouts: list[Rollout]) -> dict[str, list[Rollout]]:
        grouped_rollouts: dict[str, list[Rollout]] = {}
        for rollout in rollouts:
            grouped_rollouts.setdefault(rollout.group_id, []).append(rollout)
        return grouped_rollouts

    def _build_group_payload(self, group: list[Rollout]) -> RolloutSample:
        first_rollout = group[0]
        output = RolloutSample(
            item=[first_rollout.item] * len(group),
            prompt_ids=list(first_rollout.prompt_ids),
            completions=[],
            completion_ids=[],
            completion_logprobs=[],
            stop_reason=[],
            finish_reason=[],
            epoch=first_rollout.epoch,
            inputs=first_rollout.prompt,
        )
        for rollout in group:
            output.completions.append(rollout.completion)
            output.completion_ids.append(rollout.completion_ids)
            output.completion_logprobs.append(rollout.completion_logprobs)
            output.stop_reason.append(rollout.stop_reason)
            output.finish_reason.append(rollout.finish_reason)
        return output

    def should_keep_sample(self, sample: RolloutSample) -> bool:
        return sample.std != 0.0

    def _filter_payload_samples(self, samples: list[RolloutSample]) -> list[RolloutSample]:
        return [sample for sample in samples if self.should_keep_sample(sample)]
    
    def payload(
        self,
        rollouts: list[Rollout] | Rollout,
        calculate_rewards: bool = True,
        tokenizer: AutoTokenizer | None = None,
    ) -> list[RolloutSample]:
        rollouts = [rollouts] if isinstance(rollouts, Rollout) else rollouts

        samples: list[RolloutSample] = []
        grouped_rollouts = self._group_rollouts(rollouts)
        for group in grouped_rollouts.values():
            output = self._build_group_payload(group)
            if calculate_rewards and self.reward_functions:
                output.all_rewards, output.rewards, output.mean, output.std = (
                    self.calculate_rewards(
                        output.item,
                        output.completions,
                        output.completion_ids,
                        group,
                        tokenizer=tokenizer,
                    )
                )
            else:
                output.all_rewards, output.rewards, output.mean, output.std = {}, [], 0.0, 0.0

            samples.append(output)
        return self._filter_payload_samples(samples)

    def calculate_rewards(self, items, completions, completion_ids, rollouts, tokenizer: AutoTokenizer | None = None):
        """Calculate rewards using the environment's reward functions."""
        all_rewards = {}
        if not self.reward_functions:
            return {}, [], 0.0, 0.0

        reward_meta = {}
        needs_cleaned = False
        for reward_function in self.reward_functions:
            sig = inspect.signature(reward_function)
            params = sig.parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            reward_meta[reward_function] = (params, accepts_kwargs)
            if accepts_kwargs or "cleaned_completions" in params:
                needs_cleaned = True

        cleaned_completions = None
        if needs_cleaned:
            cleaned_completions = clean_completions(
                completions,
                tokenizer=tokenizer,
                token_ids_list=completion_ids,
            )

        for reward_function in self.reward_functions:
            params, accepts_kwargs = reward_meta[reward_function]
            base_kwargs = {
                "rollouts": rollouts,
                "items": items,
                "completions": completions,
                "completion_ids": completion_ids,
            }
            extra_kwargs = {}
            if tokenizer is not None:
                extra_kwargs["tokenizer"] = tokenizer
            if cleaned_completions is not None:
                extra_kwargs["cleaned_completions"] = cleaned_completions

            if accepts_kwargs:
                kwargs = {**base_kwargs, **extra_kwargs}
            else:
                combined = {**base_kwargs, **extra_kwargs}
                kwargs = {k: v for k, v in combined.items() if k in params}
            if kwargs:
                rewards = reward_function(**kwargs)
            else:
                rewards = reward_function(rollouts)
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

        self._enforce_context_limit(rollouts, process_kwargs)
                
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
