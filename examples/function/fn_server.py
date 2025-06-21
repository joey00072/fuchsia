from fuchsia.vllm_server import DataSamplerServer, ServerConfig
from datasets import load_dataset
from rich import print
from typing import Optional
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path    
import re
from fuchsia.envs import MultiTurnEnvironment, Rollout
from dataclasses import dataclass
import json

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def response_format_reward(rollouts: list[Rollout], *args, **kwargs) -> list[float]:
    # correct_answer = sample["correct_answer"]
    lst = []  
    for rollout in rollouts:
        print(rollout.state["reward"])
        lst.append(rollout.state["reward"])
    return lst


    

def prepare_dataset(dataset, tokenizer) -> Dataset:

    def process_example(example: dict) -> Optional[dict]:
        prefix = """Online function calling is avalible while thinking in <think> tag 
[
    {
        "name": "python_interpreter",
        "description": "Python interpreter that takes code string as input and returns the output.",
        "parameters": {
        "code": {
                "description": "The code to execute. only std output will be returned.",
                "type": "str",
                "default": ""
                }
        }
    }
]\n\n

give answer in  explanation \n<answer>number</answer> format.
"""
        example["text"] = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prefix + example["prompt"][0]["content"]},
            ],
            tokenize=False,
        )
        return example

  
    dataset = dataset.map(
        process_example,
        desc="Processing dataset",
    )
    return dataset

@dataclass
class PythonInterpreterEnvironment(MultiTurnEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = ["</request>"]
        self.reward_functions = [response_format_reward]

    def step_rollout(self, rollout: Rollout):
        rollout.state["reward"] = 0
        if rollout.stop_reason != "</request>":
            return
        
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
        code = rollout.completion
        
        try:
            code = code.split("<request>")[1].strip()
            code = json.loads(code)
            code = code["arguments"]["code"]
            output = python(code)
        except Exception as e:
            output = f"Error: {e}"
        print("Output: ", output)
        rollout.state["reward"] = 0
        if "error" in output.lower():
            rollout.state["reward"] = 0
        elif output!="":
            rollout.state["reward"] = 1
        else:
            rollout.state["reward"] = 0
            
        rollout.last_completion += "</request>\n<response>" + output + "</response> \n<function_call>"
        rollout.completion += "</request>\n<response>" + output + "</response> \n<function_call>"
        rollout.stop_reason = ""


def main():
    server_config = ServerConfig.from_yaml(Path(__file__).parent / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(server_config.model)
    dataset = load_dataset(server_config.dataset_name, server_config.dataset_split)["math"]
    dataset = prepare_dataset(dataset, tokenizer)
    
    server = DataSamplerServer(server_config, dataset, environment=PythonInterpreterEnvironment())
    server.serve()


if __name__ == "__main__":
    main()
    