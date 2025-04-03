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


prefix = "give correct answer (not option) boxed\nQuestion:"
def format_prompt(row):
    prompt = prefix + row["question"]
    if row["Type"] == "MCQ":
        options = json.loads(row["options"])
        for option in options:
            prompt += f"\n{option['identifier']}. {option['content']}"
            
    message = [
        {"role": "user", "content": prompt}
    ]
    row["text"] = tokenizer.apply_chat_template(message, tokenize=False)
    return row

dataset = load_dataset(dataset_name, split="train")


dataset = dataset.map(format_prompt)

print(dataset[0])


def extract_boxed_values(output):
    # Match \boxed{...} including nested braces using a balanced approach
    pattern = r"\\boxed\s*{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, output)
    return [match.replace('\n', '').replace('_', '').strip() for match in boxed_matches]



def test_datasampler():
    max_model_len = 8*1024
    config = DataSamplerConfig(
        model=model_name,
        revision="main",
        tensor_parallel_size=1,
        host="0.0.0.0",
        port=8000,
        dataset_feild="text",
        buffer_size=4,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.98,
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
    def reward_function_1(tokenizer, samples, completions, *args, **kwargs):
        return [len(completion)/max_model_len for completion in completions]
    
    def reward_function_2(tokenizer, samples, completions, *args, **kwargs):
        return [len(completion)/max_model_len for completion in completions]
    
    server = DataSamplerServer(config, dataset, [reward_function_1, reward_function_2])
    server.serve()  
    
if __name__ == "__main__":
    test_datasampler()