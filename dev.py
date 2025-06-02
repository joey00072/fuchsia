from dataclasses import dataclass, field
from vllm import LLM, SamplingParams

from concurrent.futures import ThreadPoolExecutor
from rich import print

# max_new_tokens = 256*2
# prompts = [
#     "The president of the United States is",
#     "10 cities in india are"
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=max_new_tokens)


# llm = LLM(model="unsloth/Llama-3.2-1B-Instruct",max_model_len=max_new_tokens)


# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Stop reason: {output.outputs[0].stop_reason} ")
#     print("Finishe reason: ", output.outputs[0].finish_reason)
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
# print("==<<<END>>==\n")






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

            
            
            
            
            
        
        
        
    
 

print(Rollout(prompt="what is 2233+3234"))