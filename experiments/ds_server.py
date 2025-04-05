from fuchsia.vllm_server import DataSamplerServer, DataSamplerConfig
from datasets import load_dataset
import json
from rich import print
import re

from transformers import AutoTokenizer

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_name = "AthenaAgent42/jee_papers"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


prefix = "give correct answer boxed\nQuestion:"


def format_prompt(row):
    prompt = prefix + row["question"]
    if row["Type"] == "MCQ":
        options = json.loads(row["options"])
        for option in options:
            prompt += f"\n{option['identifier']}. {option['content']}"

    message = [{"role": "user", "content": prompt}]
    row["text"] = tokenizer.apply_chat_template(message, tokenize=False)
    if "MCQ" in row["Type"]:
        option = row["correct_option"]
        answer = {"1":"A","2":"B","3":"C","4":"D"}[option]
    else:
        answer = row["correct_answer"]
    
    row["answer"] = answer
    return row


dataset = load_dataset(dataset_name, split="train")


dataset = dataset.map(format_prompt).filter(lambda x: int(x["pass_rate"]) <= 8)
print(dataset[0])


import regex as re
def find_boxes(text):
    pattern = r"boxed\{((?:[^{}]+|(?R))*)\}"
    matches = re.findall(pattern, text)
    return matches 

def reward_func(tokenizer, samples, completions, *args, **kwargs) -> list[float]:
    OPEN_THINK = "<think>"
    CLOSE_THINK = "</think>"
    print(completions[0])
    rewards = []
    for sample, completion in zip(samples, completions):
        reward = 0.0
        text = completion
        answer = sample["answer"]
        if OPEN_THINK in text:
            reward = 0.1
            
            if CLOSE_THINK in text:
                reward += .01
                output = text.split(CLOSE_THINK)[1]
                
                boxes = find_boxes(output)
                if len(boxes) > 0:
                    reward += 1.0
                    box_value = boxes[-1]
                    if box_value == answer:
                        reward += 5.0
                    else:
                        reward -= 1.0   
                else:
                    reward -= 1.0
            
            else:
                reward -= .01
        else:
            reward -= .01
        
        rewards.append(reward)

            
    return rewards


def test_datasampler():
    max_model_len = 1 * 1024
    config = DataSamplerConfig(
        model=model_name,
        revision="main",
        tensor_parallel_size=1,
        host="0.0.0.0",
        port=8000,
        dataset_feild="text",
        buffer_size=8,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.7,
        dtype="bfloat16",
        vllm_max_tokens=max_model_len,
        vllm_n=8,
        vllm_repetition_penalty=1.0,
        vllm_temperature=0.9,
        vllm_top_p=1.0,
        vllm_top_k=100,
        vllm_min_p=0.0,
        enable_prefix_caching=False,
        generation_batch_size=5,
        # quantization="fp8",
    )

    server = DataSamplerServer(config, dataset, [reward_func])
    server.serve()


if __name__ == "__main__":
    test_datasampler()
