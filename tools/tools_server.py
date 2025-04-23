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

from picothinking_dataset import PicoThinkingFunctionCalling, load_config
    

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"





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
        
        call_idx = 0
        while not finished:
            finished = True
            inputs = []
            call_idx += 1
            if call_idx > 10:
                break
            
            for key, value in unfinished.items():
                b_indx, g_indx = key
                prompt, text = value
                if CLOSE_THINK not in text and OPEN_FUNCTION_CALL not in text: 
                    patch = ( 
                             OPEN_FUNCTION_CALL
                             +OPEN_REQUEST
                             +items[b_indx]["request"]
                             +CLOSE_REQUEST
                            +OPEN_RESPONSE
                            +items[b_indx]["response"]
                            +CLOSE_RESPONSE
                            +CLOSE_FUNCTION_CALL
                            )
                else:
                    patch = OPEN_FUNCTION_CALL
                unfinished[key] = (prompt, text+patch)
            
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
        if completion.strip().startswith(OPEN_THINK):
            reward += 0.3
            if completion.count(OPEN_THINK)==1:
                reward += 0.2

        if CLOSE_THINK in completion and completion.count(CLOSE_THINK) == 1:
            thinking, response = completion.split(CLOSE_THINK)
            tokens_in_think = []
            for tok in SPECIAL_TOKENS:
                if tok in thinking:
                    reward += 0.3
                    tokens_in_think.append(tok)
                if tok in response:
                    reward -= 0.2
            
            if all(tok in tokens_in_think for tok in SPECIAL_TOKENS) and len(tokens_in_think) == len(SPECIAL_TOKENS):
                reward += 0.3
        else:
            reward -= 0.0
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
        vllm_repetition_penalty=config["server"]["vllm"].get("repetition_penalty", 0.9),
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
