import concurrent.futures
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Optional

from datasets import Dataset, load_dataset
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from numpy import roll
# from rich import print
from transformers import AutoTokenizer

from fuchsia.envs import MultiTurnEnvironment, Rollout
from fuchsia.vllm_server import DataSamplerServer, ServerConfig

import multiprocessing
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
# combined.py
import asyncio
from fastmcp import Client




# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def find_last_box_value(text):
    """
    Finds all occurrences of \boxed{...} in the given text,
    and returns the content of the last box.
    Returns None if no box is found.
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if not matches:
        return None
    last = matches[-1]
    try:
        return int(last)
    except ValueError:
        return last

def single_rollout_reward(rollout: Rollout) -> float:
    reward = 0
    completion = rollout.completion
    print(completion)
    print("--------------------------------")
    format_reward = 0
    correctness_reward = 0
    if "<think>" not in completion or completion.count("<think>") != 1:
        return 0
    format_reward += 0.1
    if "</think>" in completion and completion.count("</think>") == 1:
        format_reward += 0.1
        think, output = completion.split("</think>")
        if "<tool_call>" not in think:
            return format_reward
        format_reward += 0.1
        if "</tool_call>" in completion:
            return format_reward
        format_reward += 0.1
        answer = find_last_box_value(output)
        correct_answer = rollout.item["answer"]
        try:
            if int(correct_answer) == correct_answer:
                correct_answer = int(correct_answer)
            else:
                correct_answer = correct_answer
        except:
            correct_answer = correct_answer
        
        if answer is not None and str(answer) == str(correct_answer):
            correctness_reward = 5
        else:
            correctness_reward = 0.3
    else:
        correctness_reward = 0
        
    return format_reward + correctness_reward

def response_format_reward(rollouts: list[Rollout], *args, **kwargs) -> list[float]:
    # correct_answer = sample["correct_answer"]
    lst = []  
    for rollout in rollouts:
        reward = single_rollout_reward(rollout)
        lst.append(reward)
    return lst


    

def prepare_dataset(dataset, tokenizer) -> Dataset:

    def process_example(example: dict) -> Optional[dict]:
        prefix = """Online function calling is avalible while in thinking:\n [
    {
        "name": "ipython_interpreter",
        "description": "ipython interpreter that takes code string as input and returns the output.",
        "parameters": {
        "code": {
                "description": "The code to execute. only std output will be returned.",
                "type": "str",
                "default": ""
                }
        }
    }
]


"""
        example["text"] = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prefix + example["problem"] +"\n give answer in \\boxed{number} format."},
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
        self.max_steps = 4
        self.stop = ["</tool_call>"]
        self.reward_functions = [response_format_reward]

    def step_rollout(self, rollout: Rollout):
        rollout.state["reward"] = 0
        if rollout.stop_reason != "</tool_call>":
            return
        code = rollout.completion
        
        try:
            code = code.split("<tool_call>")[1].strip()
            code = json.loads(code)
            code = code["arguments"]["code"]
            async def async_run():
                async with Client(
                    "http://localhost:8111/mcp",
                    ) as client:
                    output = await client.call_tool("execute", {"code": code})
                    return output
                    # output = output["stdout/stderr"] 
            output = asyncio.run(async_run()).data
            print("FUCK YEAH!!!")
        except Exception as e:
            output = {"stdout/stderr": f"Error: {e}"}
        print("Output: ", output)
        output = output["stdout/stderr"]
        
        rollout.last_completion += "</tool_call>\n<tool_response>\n" + output + "\n</tool_response>"
        rollout.completion += "</tool_call>\n<tool_response>\n" + output + "\n</tool_response>"
        rollout.stop_reason = ""


def main():
    server_config = ServerConfig.from_yaml(Path(__file__).parent / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(server_config.model)
    dataset = load_dataset(server_config.dataset_name, server_config.dataset_split)["train"]
    dataset = prepare_dataset(dataset, tokenizer)
    
    server = DataSamplerServer(server_config, dataset, environment=PythonInterpreterEnvironment())
    server.serve()


if __name__ == "__main__":
    main()
    