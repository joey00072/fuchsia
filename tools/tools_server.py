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
from vllm import LLM, SamplingParams
from datasets import load_dataset
from rich import print




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[red]Failed to load config from {config_path}: {e}[/red]")
        raise



OPEN_FUNCTION_CALL = "<function_call>"
CLOSE_FUNCTION_CALL = "</function_call>"
OPEN_REQUEST = "<request>"
CLOSE_REQUEST = "</request>"
OPEN_RESPONSE = "<response>"
CLOSE_RESPONSE = "</response>"
OPEN_THINK = "<think>"
CLOSE_THINK = "</think>"

SPECIAL_TOKENS = [
    OPEN_FUNCTION_CALL,
    CLOSE_FUNCTION_CALL,
    OPEN_REQUEST,
    CLOSE_REQUEST,
    OPEN_RESPONSE,
    CLOSE_RESPONSE,
]


prefix = """Online function calling is avalible while thinking.
function call format:
<function_call>
<request>
...
</request>
<response>
...
</response>
</function_call>
Available functions:

"""




class PicoThinkingFunctionCalling:
    def __init__(self, tokenizer=None):
        seed_dataset_name = "joey00072/pico_thinking_function_calling"
        self.seed_dataset = load_dataset(seed_dataset_name)["train"]
        def prepare(example):
            prompt = prefix+example["schema"]+"\n\n"+example["question"]
            example["tools"] = example["schema"]
            if tokenizer is not None:
                messages = [
                            {
                                "role": "system",
                                "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
            return example
        
        self.seed_dataset = self.seed_dataset.map(prepare)
        main_dataset_name = "Salesforce/xlam-function-calling-60k"
        self.main_dataset = load_dataset(main_dataset_name)["train"]
        def prepare(example):
            prompt = prefix+example["tools"]+"\n\n"+example["query"]
            if tokenizer is not None:
                messages = [
                            {
                                "role": "system",
                                "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False)
            return example
        self.main_dataset = self.main_dataset.map(prepare)
        
        
        self._seed_len = len(self.seed_dataset)
        self._main_len = len(self.main_dataset)
        
    def __len__(self):
        return self._seed_len + self._main_len
    
    def __iter__(self):
        for item in self.seed_dataset:
            yield item
        for item in self.main_dataset:
            yield item

    def __getitem__(self, idx):
        if idx < self._seed_len:
            return self.seed_dataset[idx]
        else:
            return self.main_dataset[idx - self._seed_len]
        
    def shuffle(self, seed=42):
        self.seed_dataset = self.seed_dataset.shuffle(seed=seed)
        self.main_dataset = self.main_dataset.shuffle(seed=seed)
        return self
    



class ToolsSamplerServer(DataSamplerServer):
    
    def __init__(self, *args, **kwargs):
        kwargs["pre_fill_buffer"] = False
        super().__init__(*args, **kwargs)
        self.buffer_fill()
        
    def process_sample(self, items):
        TOOLS_CALL_TOKEN = OPEN_FUNCTION_CALL
        prompts = [item[self.dataset_field] for item in items]
        sampling_params = SamplingParams(
            n=self.config.vllm_n,
            repetition_penalty=self.config.vllm_repetition_penalty,
            temperature=self.config.vllm_temperature,
            top_p=self.config.vllm_top_p,
            top_k=self.config.vllm_top_k,
            min_p=self.config.vllm_min_p,
            max_tokens=self.config.vllm_max_tokens,
            stop=[TOOLS_CALL_TOKEN]
        )
        output_buffer = [["" for _ in range(self.config.vllm_n)] for _ in range(len(items))]
        stop_reason_buffer = [[None for _ in range(self.config.vllm_n)] for _ in range(len(items))]
        finished_buffer = [[None for _ in range(self.config.vllm_n)] for _ in range(len(items))]
        
        import time
        start_time = time.perf_counter()
        vllm_outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time} seconds")


        sampling_params = SamplingParams(
            n=1,
            repetition_penalty=self.config.vllm_repetition_penalty,
            temperature=self.config.vllm_temperature,
            top_p=self.config.vllm_top_p,
            top_k=self.config.vllm_top_k,
            min_p=self.config.vllm_min_p,
            max_tokens=self.config.vllm_max_tokens,
            stop=[TOOLS_CALL_TOKEN]
        )
        finished = True
        unfinished = {}
        stop_reason_check = {}
        finished_check = {}
        
        for sidx, (prompt, outputs) in enumerate(zip(prompts, vllm_outputs)):
            for gidx, output in enumerate(outputs.outputs):
                if output.stop_reason == TOOLS_CALL_TOKEN:
                    finished = False
                    unfinished[(sidx, gidx)] = (prompt, output.text)
                else:
                    output_buffer[sidx][gidx]+=output.text
                stop_reason_check[(sidx, gidx)] = output.stop_reason
                finished_check[(sidx, gidx)] = output.finish_reason
                
        while not finished:
            finished = True
            inputs = []
            
            for key, value in unfinished.items():
                prompt, text = value
                unfinished[key] = (prompt, text+TOOLS_CALL_TOKEN)
            
            for key, value in unfinished.items():
                prompt, text = value
                inputs.append(prompt+text)
                
            vllm_outputs = self.llm.generate(inputs, sampling_params=sampling_params)
            finished_idx = []
            for idx,(key, value)in enumerate(unfinished.items()):
                output =  vllm_outputs[idx].outputs[0]
                if output.stop_reason == TOOLS_CALL_TOKEN:
                    finished = False
                    prompt, text = value
                    text+=output.text
                    unfinished[key] = (prompt, text)
                else:
                    x, y = key
                    output_buffer[x][y]+=value[1]+output.text
                    finished_idx.append(key)
                stop_reason_check[key] = output.stop_reason
                finished_check[key] = output.finish_reason
                
            for idx in finished_idx:
                unfinished.pop(idx)
                
        for x, y in stop_reason_check.keys():
            stop_reason_buffer[x][y] = stop_reason_check[(x, y)]
        for x, y in finished_check.keys():
            finished_buffer[x][y] = finished_check[(x, y)]
            
        print(f"Total Time taken: {time.perf_counter() - start_time} seconds")
        completions = []
        for output in output_buffer:
            for o in output:
                completions.append(o)
                # print("+"*100)
                # print(o)
                # print("-"*100)

        completion_ids = [
            list(output.token_ids)
            for outputs in vllm_outputs
            for output in outputs.outputs
        ]
        
        

        all_outputs = []
        for g_idx, (item,g_completion) in enumerate(zip(items,output_buffer)):
            print(item)
            output = {}
            output["item"] = [item] * self.config.vllm_n
            output["inputs"] = [item[self.dataset_field] for item in items]
            output["completions"] = g_completion
            if "text" in output["item"][0]:
                output["completions"][0] = "<think>"+output["item"][0]["text"].split("<think>")[1]
            output["completion_ids"] = [self.tokenizer.encode(c) for c in output["completions"]]
            output["stop_reason"] = stop_reason_buffer[g_idx]
            output["finish_reason"] = finished_buffer[g_idx]
            output["epoch"] = self._epoch
            
            for idx in range(self.config.vllm_n):
            #     g_completion = completions[g_idx * self.config.vllm_n + idx]
            #     g_completion_id = completion_ids[g_idx * self.config.vllm_n + idx]
            #     g_stop_reason = stop_reason[g_idx * self.config.vllm_n + idx]
            #     g_finish_reason = finish_reason[g_idx * self.config.vllm_n + idx]
                
                output["inputs"] = item[self.dataset_field]
            #     output["completions"].append(g_completion)
            #     output["completion_ids"].append(g_completion_id)
            #     output["stop_reason"].append(g_stop_reason)
            #     output["finish_reason"].append(g_finish_reason)

            output["all_rewards"], output["rewards"], output["mean"], output["std"] = (
                self.calculate_rewards(
                    output["item"], output["completions"], output["completion_ids"]
                )
            )
            all_outputs.append(output)
        return all_outputs
    
    
def reward_func(tokenizer, samples, completions, *args, **kwargs) -> list[float]:
    rewards = []
    for sample, completion in zip(samples, completions):
        reward = 0.0
        if CLOSE_THINK in completion and completion.count(CLOSE_THINK) == 1:
            thinking, response = completion.split(CLOSE_THINK)
            for tok in SPECIAL_TOKENS:
                if tok in thinking:
                    reward += 0.3
                if tok in response:
                    reward -= 0.2
                if "```" in response:
                    reward -= 0.1
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def test_datasampler():
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Initialize model and tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    

    dataset = PicoThinkingFunctionCalling(tokenizer=tokenizer)

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
        
        dataset_field = config["dataset"]["field"],
        # VLLM specific parameters
        vllm_n=config["server"]["vllm"]["n"],
        vllm_temperature=config["server"]["vllm"]["temperature"],
        vllm_top_p=config["server"]["vllm"]["top_p"],
        vllm_top_k=config["server"]["vllm"]["top_k"],
        vllm_min_p=config["server"]["vllm"]["min_p"],
        vllm_max_tokens=config["server"]["vllm"]["max_tokens"],
        vllm_repetition_penalty=config["server"]["vllm"].get("repetition_penalty", 1.0),
        vllm_kv_quantization=config["server"]["vllm"].get("kv_quantization", False),
        
    )
    
    print(server_config )
    # Create and start server
    print("[blue]Starting server...[/blue]")
    server = ToolsSamplerServer(
        config=server_config,
        dataset=dataset,
        reward_functions=[reward_func],
    )
    server.serve()


if __name__ == "__main__":
    test_datasampler()
