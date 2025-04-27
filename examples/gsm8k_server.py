from fuchsia.vllm_server import DataSamplerServer, ServerConfig
from datasets import load_dataset
import json
from rich import print
import re
from typing import Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "unsloth/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
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
        required_tags = [
            START_THINKING_TOKEN,
            END_THINKING_TOKEN,
            START_ANSWER_TOKEN,
            END_ANSWER_TOKEN,
        ]
        for tag in required_tags:
            if tag in s:
                format_reward += 0.15
                if s.count(tag) > 1:
                    format_reward -= s.count(tag) * 0.01

        # Validate thinking section
        if s.count("<thinking>") == 1:
            format_reward += 0.5
            thinking_content = (
                s.split(START_THINKING_TOKEN)[1].split(END_THINKING_TOKEN)[0].strip()
            )
            if len(thinking_content) > 10:  # Basic content validation
                content_reward += 0.5
        else:
            format_reward -= 0.1

        # Validate answer section
        if "<answer>" in s and "</answer>" in s:
            format_reward += 0.4
            answer_content = (
                s.split(START_ANSWER_TOKEN)[1].split(END_ANSWER_TOKEN)[0].strip()
            )
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


def prepare_dataset(dataset, tokenizer=tokenizer) -> Dataset:
    """Prepare the GSM8K dataset with better error handling and validation."""

    def extract_hash_answer(text: str) -> Optional[str]:
        try:
            if "####" not in text:
                return None
            answer = text.split("####")[1].strip()
            # Validate that the answer is a number
            answer = answer.replace(",", "").strip()
            answer = float(answer)
            return answer
        except (ValueError, IndexError):
            return None

    def process_example(example: dict) -> Optional[dict]:
        # print(f"{example=}")
        try:
            answer = extract_hash_answer(example["answer"])
            example["correct_answer"] = answer
            if answer is None:
                print("FUCKKKKKKKKK.")
                print(f"{example['answer']=}")
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


def load_config(config_path: str = "examples/gsm8k_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_datasampler():
    config = load_config()
    
    server_config = ServerConfig(
        model=config['model']['name'],
        host=config['server']['host'],
        port=config['server']['port'],
        dataset_field=config['dataset']['field'],
        buffer_size=config['server']['buffer_size'],
        max_model_len=config['model']['max_model_len'],
        gpu_memory_utilization=config['server']['gpu_memory_utilization'],
        dtype=config['model']['dtype'],
        vllm_max_tokens=config['server']['vllm']['max_tokens'],
        vllm_n=config['server']['vllm']['n'],
        vllm_temperature=config['server']['vllm']['temperature'],
        vllm_top_p=config['server']['vllm']['top_p'],
        vllm_top_k=config['server']['vllm']['top_k'],
        vllm_min_p=config['server']['vllm']['min_p'],
        enable_prefix_caching=config['server']['enable_prefix_caching'],
        generation_batch_size=config['server']['generation_batch_size'],
        quantization=config['server']['quantization'],
    )
    
    dataset = load_dataset(config['dataset']['name'], config['dataset']['split'])["train"]
    print(dataset)
    # if config['dataset']['max_samples']:
    #     dataset = dataset.select(range(min(config['dataset']['max_samples'], len(dataset))))

    
    dataset = prepare_dataset(dataset)
    
    server = DataSamplerServer(server_config, dataset, [reward_function_1])
    server.serve()


if __name__ == "__main__":
    test_datasampler()
