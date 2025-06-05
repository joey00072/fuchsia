from dataclasses import dataclass, field
from vllm import LLM, SamplingParams

import numpy as np

from concurrent.futures import ThreadPoolExecutor
from rich import print


@dataclass
class Rollout:
    prompt: str = ""
    completion: str = ""
    last_completion: str = ""
    stop: list[str] = field(default_factory=list)
    stop_reason: str = ""
    finish_reason: str = ""
    completed: bool = False
    state:dict = field(default_factory=dict)
    
    def __post_init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property        
    def is_completed(self):
        return self.finish_reason in ["stop", "length"]
    
    def input(self):
        return self.prompt+self.last_completion
    
    
@dataclass
class Environment:
    def process_rollouts(self, rollouts: list[Rollout]):
        for rollout in rollouts:
            if rollout.finish_reason in ["stop", "length"]:
                rollout.completed = True
        return rollouts
    

class MultiTurnEnvironment(Environment):
    def process_rollouts(self, rollouts: list[Rollout]):
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
                ['python', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
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

    
class RolloutManager:
    def __init__(self,llm: LLM, sampling_params: SamplingParams, environment: Environment = None):
        self.llm = llm
        self.sampling_params = sampling_params
        self.rollouts = []
        self.stop: list[str]|None = None
        
    def generate(self, prompts: list[str], n: int = 1, environment: Environment = None):
        
       rollouts = []
       for prompt in prompts:
           rollouts.extend([Rollout(prompt=prompt) for _ in range(n)])
               
       while any(not rollout.completed for rollout in rollouts):
           incomplete_rollouts = [rollout for rollout in rollouts if not rollout.completed]
           self._generate(incomplete_rollouts, environment)


        # while all rollouts are not completed
            # filter incomplete rollouts
            # genrate with vllm
            # pass it to environment
            # update rollouts
            
        # return rollouts
        
    def _generate(self, rollouts: list[Rollout], environment: Environment):
       
       inputs = []
       for rollout in rollouts:
           inputs.append(rollout.prompt+rollout.completion)
           
       completions = self.llm.generate(
           inputs,
           sampling_params=self.sampling_params,
           use_tqdm=False,
           stop=self.stop,
       )
       rollouts = environment.process_rollouts(rollouts)
       return rollouts

            
            
    def calculate_rewards(self, items, completions, completion_ids):
        """Calculates rewards for generated completions."""
        if not self.is_data_sampler:
            return {}, [], 0.0, 0.0

        all_rewards = {}
        for reward_function in self.reward_functions:
            rewards = reward_function(
                self.tokenizer, items, completions, completion_ids
            )
            all_rewards[reward_function.__name__] = rewards

        reward_values = np.array([list(rewards) for rewards in all_rewards.values()])
        total_rewards = reward_values.sum(axis=0)
        mean = float(total_rewards.mean())
        std = float(total_rewards.std())

        return all_rewards, total_rewards.tolist(), mean, std
            
            
        
        
        
    
 

print(Rollout(prompt="what is 2233+3234"))




