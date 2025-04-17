from fuchsia.vllm_server import DataSamplerServer, ServerConfig
from datasets import load_dataset
import json
from rich import print
import re
from typing import Optional
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from pathlib import Path
import regex
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[red]Failed to load config from {config_path}: {e}[/red]")
        raise

# Initialize tokenizer and model after config is loaded
def initialize_model(config):
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cpu()
    return tokenizer, model

prefix = "give correct option/answer boxed\nQuestion:"

def format_prompt(row):
    prompt = prefix + row["question"]
    if row["Type"] == "MCQ":
        options = json.loads(row["options"])
        for option in options:
            prompt += f"\n{option['identifier']}. {option['content']}"

    message = [
        {
        "role": "system",
        "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
        },
        {"role": "user", "content": prompt}
        ]
    row["text"] = tokenizer.apply_chat_template(message, tokenize=False)
    if "MCQ" in row["Type"]:
        option = row["correct_option"]
        answer = {"1":"A","2":"B","3":"C","4":"D"}[option]
    else:
        answer = row["correct_answer"]
    
    row["answer"] = answer
    return row

def find_boxes(text):
    pattern = r"boxed\{((?:[^{}]+|(?R))*)\}"
    matches = regex.finditer(pattern, text)
    boxes = []
    for match in matches:
        boxes.append(match.group(1))
    return boxes

def reward_func(tokenizer, samples, completions, *args, **kwargs) -> list[float]:
    rewards = []
    for sample, completion in zip(samples, completions):
        reward = 0.0
        boxes = find_boxes(completion)
        if "boxed{" in completion:
            reward += 0.001
        if not boxes:
            rewards.append(reward)
            continue
        last_box = boxes[-1]
        if last_box.strip() == sample["answer"]:
            reward += 10.0
        else:
            reward -= 0.0
            
        rewards.append(reward)
    max_completion_len = max(len(completion) for completion in completions)
    for idx,(reward,completion) in enumerate(zip(rewards,completions)):
        rewards[idx] = reward + (1 - (len(completion) / max_completion_len)) *0.1
    return rewards

def test_datasampler():
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Initialize model and tokenizer
    global tokenizer
    tokenizer, model = initialize_model(config)
    
    # Load and prepare dataset
    print("[blue]Loading dataset...[/blue]")
    dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    print(f"[green]Loaded dataset with {len(dataset)} examples[/green]")
    
    # Format prompts and filter
    print("[blue]Formatting prompts...[/blue]")
    dataset = dataset.map(format_prompt)
    # dataset = dataset.filter(lambda x: int(x["pass_rate"]) <= 12)
    print(f"[green]Filtered dataset to {len(dataset)} examples[/green]")
    
    # Create server config
    server_config = ServerConfig(
        model=config["model"]["name"],
        host=config["server"]["host"],
        port=config["server"]["port"],
        gpu_memory_utilization=config["server"]["gpu_memory_utilization"],
        tensor_parallel_size=config["server"]["tensor_parallel_size"],
        enable_prefix_caching=config["server"]["enable_prefix_caching"],
        buffer_size=config["server"]["buffer_size"],
        generation_batch_size=config["server"]["generation_batch_size"],
        quantization=config["server"]["quantization"],
        max_model_len=config["model"]["max_model_len"],
        
        # VLLM specific parameters
        vllm_n=config["server"]["vllm"]["n"],
        vllm_temperature=config["server"]["vllm"]["temperature"],
        vllm_top_p=config["server"]["vllm"]["top_p"],
        vllm_top_k=config["server"]["vllm"]["top_k"],
        vllm_min_p=config["server"]["vllm"]["min_p"],
        vllm_max_tokens=config["server"]["vllm"]["max_tokens"],
        vllm_repetition_penalty=config["server"]["vllm"].get("repetition_penalty", 1.0)
    )
    
    # Create and start server
    print("[blue]Starting server...[/blue]")
    server = DataSamplerServer(
        config=server_config,
        dataset=dataset,
        reward_functions=[reward_func],
    )
    server.serve()


if __name__ == "__main__":
    test_datasampler()
