
from vllm import LLM,SamplingParams
from transformers import AutoTokenizer

import os
from rich import print


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = "NousResearch/DeepHermes-3-Llama-3-3B-Preview"



llm = LLM(
    model=model_name,
    revision="main",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.2,
    dtype="bfloat16",
    enable_prefix_caching=False,
    max_model_len=1024,
    # worker_cls="fuchsia.vllm_server.WeightSyncWorker",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

n = 8
sampling_params = SamplingParams(
    n=n,
    temperature=0.6,
    top_p=0.9,
    max_tokens=1024,
    stop_token=["<function_call>","</think>"]
)
text = """give correct answer boxed
Question:A line with direction cosines proportional to $$2,1,2$$ meets each of the lines $$x=y+a=z$$ and $$x+a=2y=2z$$  . The co-ordinates of each of the points of intersection are given by :
A. $$\left( {2a,3a,3a} \right),\left( {2a,a,a} \right)$$ 
B. $$\left( {3a,2a,3a} \right),\left( {a,a,a} \right)$$ 
C. $$\left( {3a,2a,3a} \right),\left( {a,a,2a} \right)$$ 
D. $$\left( {3a,3a,3a} \right),\left( {a,a,a} \right)$$"""
messages = [
            {
                "role": "system",
                "content": "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
            },
            {
                "role": "user",
                "content": text
            }
        ]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)


outputs = llm.generate(prompt, sampling_params)

for output in outputs:
    print("="*10)
    print(output.text)
    print("-"*10)
    print(output.finish_reason)
    print("="*10)




    

