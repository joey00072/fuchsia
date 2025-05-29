from fuchsia.vllm_server import ServerConfig, DataSamplerServer
from datasets import load_dataset
from vllm import LLM
from rich import print
import time

if __name__ == "__main__":
    server_config = ServerConfig(
        model="unsloth/Llama-3.2-3B-Instruct",
        host="0.0.0.0", 
        port=8000,
        dataset_field="text",
        buffer_size=4,
        max_model_len=128,
        gpu_memory_utilization=0.70,
        dtype="bfloat16",
        vllm_max_tokens=128,
        vllm_n=8,
        vllm_temperature=0.6,
        vllm_top_p=1.0,
        vllm_top_k=-1,
        vllm_min_p=0.0,
        enable_prefix_caching=False,
        generation_batch_size=4,
        quantization="bitsandbytes",
        lora_path="./lora_weights",
        single_gpu=True,
    )


    # server = DataSamplerServer(server_config, dataset, [reward_function_1])
    # server.serve()

    print("VLLM ARGS:")
    print(f"{server_config.model=}")
    print(f"{server_config.revision=}")
    print(f"{server_config.tensor_parallel_size=}")
    print(f"{server_config.gpu_memory_utilization=}")
    print(f"{server_config.dtype=}")
    print(f"{server_config.enable_prefix_caching=}")
    print(f"{server_config.max_model_len=}")

    llm = LLM(
        model=server_config.model,
        revision=server_config.revision,
        quantization=server_config.quantization,
        tensor_parallel_size=server_config.tensor_parallel_size,
        gpu_memory_utilization=server_config.gpu_memory_utilization,
        # dtype=server_config.dtype,
        enable_prefix_caching=server_config.enable_prefix_caching,
        max_model_len=server_config.max_model_len,
        enable_lora=server_config.single_gpu,
        enable_sleep_mode=True,  # Enable sleep mode for CUDA
    )

    print("Putting LLM to sleep...")
    llm.sleep(levle=1)
    print("LLM is now sleeping")

    print("Waiting 10 seconds...")
    time.sleep(10)

    print("Waking up LLM...")
    llm.wake_up()
    print("LLM is now awake")
