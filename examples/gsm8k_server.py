from fuchsia.vllm_server import DataSamplerServer, DataSamplerConfig
from datasets import load_dataset
import json
from rich import print
import re
from typing import Optional
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "unsloth/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cpu()
SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"

def response_format_reward(sample: dict, s: str, *args, **kwargs) -> float:
    """Improved reward function with better validation and scoring."""
    END_OF_TEXT_TOKEN = "<|eot_id|>"
    START_HEADER_TOKEN = "<|start_header_id|>"
    END_HEADER_TOKEN = "<|end_header_id|>"
    ASSISTANT_TOKEN = "assistant"
    USER_TOKEN = "user"
    
    START_THINKING_TOKEN = "<thinking>"
    END_THINKING_TOKEN = "</thinking>"
    START_ANSWER_TOKEN = "<answer>"
    END_ANSWER_TOKEN = "</answer>"
    
    try:
        # Extract the actual response
        try:
            s = s.split(f"{START_HEADER_TOKEN}{ASSISTANT_TOKEN}{END_HEADER_TOKEN}")[1]
        except IndexError:
            return -1.0

        if END_OF_TEXT_TOKEN in s:
            s = s.split(END_OF_TEXT_TOKEN)[0]

        # Initialize reward components
        format_reward = 0.0
        content_reward = 0.0
        correct_template = 0

        # Check format tags
        required_tags = [START_THINKING_TOKEN, END_THINKING_TOKEN, START_ANSWER_TOKEN, END_ANSWER_TOKEN]
        for tag in required_tags:
            if tag in s:
                format_reward += 0.15
                if s.count(tag) > 1:
                    format_reward -= s.count(tag) * 0.01

        # Validate thinking section
        if s.count("<thinking>") == 1:
            format_reward += 0.5
            thinking_content = s.split(START_THINKING_TOKEN)[1].split(END_THINKING_TOKEN)[0].strip()
            if len(thinking_content) > 10:  # Basic content validation
                content_reward += 0.5
        else:
            format_reward -= 0.1

        # Validate answer section
        if "<answer>" in s and "</answer>" in s:
            format_reward += 0.4
            answer_content = s.split(START_ANSWER_TOKEN)[1].split(END_ANSWER_TOKEN)[0].strip()
            try:
                answer_value = float(answer_content)
                content_reward += 1.0
                if answer_value == float(sample["answer"]):
                    content_reward += 2.0
                    correct_template += 1
            except ValueError:
                content_reward -= 0.1

        # Bonus for perfect format
        if correct_template == 1:
            format_reward += 2.0

        return format_reward + content_reward

    except Exception as e:
        print(f"[yellow]Error in reward calculation: {e}[/yellow]")
        return -1.0

def reward_function_1(tokenizer, samples, completions, *args, **kwargs):
    lst = []
    for sample, completion in zip(samples, completions):
        lst.append(response_format_reward(sample, completion))
    return lst

def prepare_dataset(dataset) -> Dataset:
    """Prepare the GSM8K dataset with better error handling and validation."""
    def extract_hash_answer(text: str) -> Optional[str]:
        try:
            if "####" not in text:
                return None
            answer = text.split("####")[1].strip()
            # Validate that the answer is a number
            answer = float(answer)
            return answer
        except (ValueError, IndexError):
            return None

    def process_example(example: dict) -> Optional[dict]:
        try:
            answer = extract_hash_answer(example["answer"])
            example["correct_answer"] = answer
            if answer is None:
                return None
            example["text"] = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["question"]},
                ],
                tokenize=False,
            )
            return example
        except Exception as e:
            print(f"[yellow]Failed to process example: {e}[/yellow]")
            return None

    try:
        dataset = dataset.map(
            process_example, 
            desc="Processing dataset",
        )
        dataset = dataset.filter(lambda x: x is not None)
        print(f"[green]Processed dataset size: {len(dataset)}[/green]")
        return dataset
    except Exception as e:
        print(f"[red]Failed to prepare dataset: {e}[/red]")
        raise


def test_datasampler():
    max_model_len = 512
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    dataset = prepare_dataset(dataset)
    config = DataSamplerConfig(
        model=model_name,
        host="0.0.0.0",
        port=8000,
        dataset_feild="text",
        buffer_size=4,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.98,
        dtype="bfloat16",
        vllm_max_tokens=max_model_len,
        vllm_n=8,  # Number of sequences to generate
        vllm_temperature=1.0,  # Temperature for sampling
        vllm_top_p=1.0,  # Nucleus sampling parameter
        vllm_top_k=-1,  # Top-k sampling parameter (-1 means disabled)
        vllm_min_p=0.0,  # Minimum probability threshold
        enable_prefix_caching=False,
        generation_batch_size=4,
        quantization=None,
    )
    server = DataSamplerServer(config, dataset, [reward_function_1])
    server.serve()  
    
if __name__ == "__main__":
    test_datasampler()