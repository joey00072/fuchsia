from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from rich import print


llm = LLM(
    model="unsloth/Llama-3.2-1b-Instruct",
    max_model_len=1024,
    enable_lora=True,
    gpu_memory_utilization=0.7,
    dtype="bfloat16",
)


sampling_params = SamplingParams(temperature=0, max_tokens=256, stop=["[/assistant]"])

prompts = [
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]

outputs = llm.generate(
    prompts,
    sampling_params,
)


print(outputs)