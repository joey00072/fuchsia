# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# derived from https://github.com/huggingface/trl/blob/main/trl/extras/vllm_client.py
# from original pr of binary-husky (https://github.com/binary-husky) pr https://github.com/huggingface/trl/pull/3094


import atexit
import logging
import time
from typing import Callable, Optional

import torch
from torch import nn


import requests
from requests import ConnectionError


from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from peft import PeftModel

logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(
        self,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
        init_communicator: bool = True,
    ):
        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self._base_url = f"http://{self.host}:{self.server_port}"
        self._init_communicator = init_communicator
        self._connection_timeout = connection_timeout
        
        # Try to connect with retries
        self._connect_with_retries()
        
        if init_communicator:
            self._init_communicator_with_retries()
            atexit.register(self.close_communicator)
        else:
            self.rank = 0
            self.pynccl_comm = None

    def _connect_with_retries(self):
        """Connect to server with automatic retries."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.check_server(self._connection_timeout)
                return
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying connection in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error("Failed to connect to VLLM server. Training will continue but VLLM operations may fail.")

    def _init_communicator_with_retries(self):
        """Initialize communicator with automatic retries."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.init_communicator()
                return
            except Exception as e:
                logger.warning(f"Communicator init attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying communicator init in 3 seconds...")
                    time.sleep(3)
                else:
                    logger.error("Failed to initialize communicator. Some operations may not work.")
                    self.rank = 0
                    self.pynccl_comm = None

    def _make_request(self, endpoint: str, method: str = "get", max_retries: int = 3, retry_delay: float = 2.0, **kwargs) -> dict:
        url = f"{self._base_url}/{endpoint.strip('/')}/"
        
        for attempt in range(max_retries + 1):
            try:
                if method.lower() == "get":
                    response = self.session.get(url, **kwargs)
                else:
                    response = self.session.post(url, **kwargs)

                if response.status_code == 200:
                    return response.json() if response.content else {}
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    logger.warning(f"Connection failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 1.5
                else:
                    logger.error(f"Failed to connect after {max_retries + 1} attempts: {str(e)}")
                    raise ConnectionError(f"Failed to connect to {url}: {str(e)}")
        
        # This should never be reached, but just in case
        raise ConnectionError(f"Unexpected error connecting to {url}")

    def check_server(self, total_timeout: float = 180.0, retry_interval: float = 2.0):
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `Fuchsia serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    print("Server is up!")
                    return None
            print( f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Generate completions with built-in fault tolerance.
        Automatically retries on failure and returns empty list instead of crashing.
        """
        params = {
            "prompts": prompts,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": guided_decoding_regex,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._make_request("generate", method="post", json=params, max_retries=2)
                return response["completion_ids"]
            except Exception as e:
                logger.warning(f"Generate attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    
        logger.error(f"Failed to generate completions after {max_retries} attempts. Returning empty list...")
        return []

    def init_communicator(self):
        response = self._make_request("get_tensor_parallel_size")
        tensor_parallel_size = response["tensor_parallel_size"]
        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size  # The client's rank is the last process

        self._make_request(
            "init_communicator",
            method="post",
            json={"host": self.host, "port": self.group_port, "world_size": world_size},
        )

        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Update named parameter with built-in fault tolerance.
        Automatically retries on failure and logs warnings instead of crashing.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._make_request(
                    "update_named_param",
                    method="post",
                    json={
                        "name": name,
                        "dtype": str(weights.dtype),
                        "shape": tuple(weights.shape),
                    },
                    max_retries=2
                )
                
                if self.pynccl_comm is not None:            
                    self.pynccl_comm.broadcast(
                        weights, src=self.rank, stream=torch.cuda.current_stream()
                    )
                    self.pynccl_comm.group.barrier()
                
                # Success - no need to log, this happens frequently
                return
                
            except Exception as e:
                logger.warning(f"Update parameter '{name}' attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    
        logger.error(f"Failed to update parameter '{name}' after {max_retries} attempts. Skipping...")

    def update_lora_params(self, model: PeftModel):
        
        if not isinstance(model, PeftModel):
            raise ValueError("Model is not a PeftModel")
            
        target_modules = model.peft_config["default"].target_modules
        alpha = model.peft_config["default"].lora_alpha
        r = model.peft_config["default"].r
        print(target_modules)
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data
            
        for name, param in model.named_parameters():
            if "lora" in name:
                continue
            if any(target in name for target in target_modules):
                if "bias" in name:
                    new_name = name.replace("base_model.model.","")
                    new_wights = param.data.clone() 
                    self.update_named_param(new_name, new_wights)
                    continue

                if "lora" not in name:
                    prefix = name.replace(".base_layer.weight","")
                    A_name = f"{prefix}.lora_A.default.weight"
                    B_name = f"{prefix}.lora_B.default.weight"
                    delta = (weights[B_name].data.clone() @ weights[A_name].data.clone()) * (alpha / r)
                    new_wights = param.data.clone() + delta
                    new_name = name.replace("base_model.model.","").replace(".base_layer.weight",".weight")
                    self.update_named_param(new_name, new_wights)

    def update_model_params(self, model: nn.Module,tokenizer=None, lora=False, single_gpu=False,lora_path=None):
        if single_gpu:
            # for name, param in model.named_parameters():
            #     if "lora" in name:
            #         print(f"{param.data.sum()}")
            if tokenizer is not None:
                tokenizer.save_pretrained(lora_path)
            model.save_pretrained(lora_path,adapter_name="grpo")
            return
        if lora:
            self.update_lora_params(model)
            return
        
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """Reset prefix cache with built-in fault tolerance."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._make_request("reset_prefix_cache", method="post", max_retries=2)
                logger.info("Prefix cache successfully reset")
                return
            except Exception as e:
                logger.warning(f"Reset prefix cache attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        logger.error(f"Failed to reset prefix cache after {max_retries} attempts. Continuing anyway...")

    def close_communicator(self):
        """Close communicator with built-in fault tolerance."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._make_request("close_communicator", method="post", max_retries=2)
                logger.info("Communicator successfully closed")
                return
            except Exception as e:
                logger.warning(f"Close communicator attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        logger.error(f"Failed to close communicator after {max_retries} attempts. Continuing anyway...")

    def get_sample(self) -> Optional[dict]:
        """
        Get a sample from the VLLM server with built-in fault tolerance.
        Automatically retries on failure and returns None instead of crashing.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._make_request("get_sample", method="post", max_retries=2)
                return response.get("sample")
            except Exception as e:
                logger.warning(f"Get sample attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    
        logger.error(f"Failed to get sample after {max_retries} attempts. Returning None...")
        return None

    def empty_buffer(self):
        """
        Empty the VLLM server buffer with built-in fault tolerance.
        Automatically retries on failure and logs warnings instead of crashing.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._make_request("empty_buffer", method="post", max_retries=2)
                logger.info("VLLM buffer successfully emptied")
                return result
            except Exception as e:
                logger.warning(f"Empty buffer attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    
        logger.error(f"Failed to empty buffer after {max_retries} attempts. Continuing anyway...")
        return {"empty_buffer": False, "error": "Max retries exceeded"}

    def fill_buffer(self, num_samples: int = None):
        """
        Fill the VLLM server buffer with built-in fault tolerance.
        Automatically retries on failure and logs warnings instead of crashing.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if num_samples is None:
                    result = self._make_request("buffer_fill", method="post", max_retries=2)
                else:
                    result = self._make_request(
                        "buffer_fill", method="post", json={"num_samples": num_samples}, max_retries=2
                    )
                logger.info("VLLM buffer successfully filled")
                return result
            except Exception as e:
                logger.warning(f"Buffer fill attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    
        logger.error(f"Failed to fill buffer after {max_retries} attempts. Continuing anyway...")
        return {"buffer_fill": False, "error": "Max retries exceeded"}

    def trigger_buffer_fill(self):
        """Trigger buffer fill with built-in fault tolerance."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._make_request("buffer_fill", method="post", max_retries=2)
                logger.info("Buffer fill successfully triggered")
                return result
            except Exception as e:
                logger.warning(f"Trigger buffer fill attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        logger.error(f"Failed to trigger buffer fill after {max_retries} attempts. Continuing anyway...")
        return {"buffer_fill": False, "error": "Max retries exceeded"}

    def buffer_status(
        self,
        *,
        timeout: float = 5.0,
        max_retries: int = 0,
        log_failures: bool = False,
    ):
        """
        Get VLLM buffer/sleep status.
        Returns an empty dict when status is unavailable.
        """
        try:
            return self._make_request(
                "buffer_status",
                method="get",
                max_retries=max_retries,
                timeout=timeout,
            )
        except Exception as e:
            if log_failures:
                logger.warning(f"Failed to fetch buffer status: {e}")
            else:
                logger.debug("Buffer status probe failed: %s", e)
            return {}

    def _wait_for_status(
        self,
        predicate: Callable[[dict], bool],
        timeout: float,
        poll_interval: float,
        max_poll_interval: float,
        backoff_factor: float,
    ) -> bool:
        """
        Poll buffer status with exponential backoff until `predicate` is satisfied or timeout elapses.
        """
        deadline = time.monotonic() + timeout
        current_interval = max(float(poll_interval), 0.1)
        max_interval = max(current_interval, float(max_poll_interval))

        while time.monotonic() < deadline:
            status = self.buffer_status(
                timeout=max(1.0, min(poll_interval * 2.0, 3.0)),
                max_retries=0,
                log_failures=False,
            )
            if status and predicate(status):
                return True
            time.sleep(current_interval)
            current_interval = min(current_interval * max(float(backoff_factor), 1.0), max_interval)
        return False

    def wait_for_buffer_ready(
        self,
        min_size: int = 1,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
        filling_grace_timeout: float = 600.0,
        assume_filling: bool = False,
    ) -> bool:
        """
        Wait until the rollout buffer has at least `min_size` items and is not actively filling.
        """
        def _to_float(value) -> float | None:
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        start_time = time.monotonic()
        base_deadline = start_time + timeout
        filling_deadline: float | None = (
            start_time + float(filling_grace_timeout) if assume_filling else None
        )
        current_interval = max(float(poll_interval), 0.1)
        max_interval = max(current_interval, float(max_poll_interval))
        last_fill_heartbeat: float | None = None
        last_fill_started: float | None = None
        last_generated_groups: int | None = None
        seen_fill_activity = bool(assume_filling)

        while True:
            now = time.monotonic()
            status_timeout = max(3.0, min(current_interval * 2.0, 8.0))
            status = self.buffer_status(
                timeout=status_timeout,
                max_retries=0,
                log_failures=False,
            )
            if status:
                current_size = int(status.get("current_size", 0))
                slot_claimed = bool(status.get("fill_slot_claimed", False))
                generation_in_progress = bool(status.get("generation_in_progress", False))
                fill_started = _to_float(status.get("last_fill_started_at"))
                fill_finished = _to_float(status.get("last_fill_finished_at"))
                has_unfinished_fill = (
                    fill_started is not None
                    and (fill_finished is None or fill_finished < fill_started)
                )
                is_filling = bool(
                    status.get("is_filling", False)
                    or slot_claimed
                    or has_unfinished_fill
                    or generation_in_progress
                )
                if current_size >= min_size and not is_filling:
                    return True

                if is_filling:
                    seen_fill_activity = True
                    heartbeat = status.get("last_fill_heartbeat_at")
                    try:
                        parsed_heartbeat = float(heartbeat) if heartbeat is not None else None
                    except (TypeError, ValueError):
                        parsed_heartbeat = None

                    if filling_deadline is None:
                        filling_deadline = now + float(filling_grace_timeout)
                    elif (
                        parsed_heartbeat is not None
                        and (last_fill_heartbeat is None or parsed_heartbeat > last_fill_heartbeat)
                    ):
                        filling_deadline = now + float(filling_grace_timeout)
                    if (
                        fill_started is not None
                        and (last_fill_started is None or fill_started > last_fill_started)
                    ):
                        filling_deadline = now + float(filling_grace_timeout)
                    generated_groups = status.get("last_fill_generated_groups")
                    try:
                        parsed_generated_groups = int(generated_groups)
                    except (TypeError, ValueError):
                        parsed_generated_groups = None
                    if (
                        parsed_generated_groups is not None
                        and (
                            last_generated_groups is None
                            or parsed_generated_groups > last_generated_groups
                        )
                    ):
                        filling_deadline = now + float(filling_grace_timeout)
                    last_fill_heartbeat = parsed_heartbeat
                    last_fill_started = fill_started
                    last_generated_groups = parsed_generated_groups
                else:
                    fill_completed = (
                        fill_started is not None
                        and fill_finished is not None
                        and fill_finished >= fill_started
                    )
                    if fill_completed and filling_deadline is not None:
                        # Fill worker completed but could not produce enough items; allow caller to retry.
                        return False
                    if not seen_fill_activity:
                        filling_deadline = None
                        last_fill_heartbeat = None

            if filling_deadline is not None:
                if now >= filling_deadline:
                    return False
            elif now >= base_deadline:
                return False

            time.sleep(current_interval)
            current_interval = min(current_interval * max(float(backoff_factor), 1.0), max_interval)

    def wait_until_sleeping(
        self,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
    ) -> bool:
        """
        Wait until server reports sleeping and no pending sleep/fill operations.
        """
        def _sleeping(status: dict) -> bool:
            is_sleeping = bool(status.get("is_sleeping", False))
            sleep_requested = bool(status.get("sleep_requested", False))
            is_filling = bool(status.get("is_filling", False))
            return is_sleeping and not sleep_requested and not is_filling

        return self._wait_for_status(
            predicate=_sleeping,
            timeout=timeout,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            backoff_factor=backoff_factor,
        )

    def wait_until_awake(
        self,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
    ) -> bool:
        """
        Wait until server reports awake state (not sleeping and no pending sleep request).
        """
        def _awake(status: dict) -> bool:
            is_sleeping = bool(status.get("is_sleeping", False))
            sleep_requested = bool(status.get("sleep_requested", False))
            return (not is_sleeping) and (not sleep_requested)

        return self._wait_for_status(
            predicate=_awake,
            timeout=timeout,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            backoff_factor=backoff_factor,
        )

    def sleep(self, max_retries=100, retry_sleep_time=2, max_retry_sleep_time=8):
        """
        Put the VLLM server to sleep with built-in fault tolerance.
        Automatically retries on failure and logs warnings instead of crashing.
        """ 
        for attempt in range(max_retries):
            try:
                response = self._make_request("sleep", method="post", max_retries=2)
                if response and response.get("sleep", False):
                    if not self.wait_until_sleeping(timeout=60.0, poll_interval=1.0):
                        logger.warning("Sleep acknowledged but server did not report steady sleeping state within timeout")
                    logger.info("VLLM client successfully put to sleep")
                    return response
                else:
                    logger.warning(f"Sleep attempt {attempt + 1} failed: {response}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_sleep_time)  # Wait before retry
                        retry_sleep_time = min(retry_sleep_time * 2, max_retry_sleep_time)
            except Exception as e:
                logger.warning(f"Sleep attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep_time)  # Wait before retry
                    retry_sleep_time = min(retry_sleep_time * 2, max_retry_sleep_time)
                    
        logger.error(f"Failed to put VLLM client to sleep after {max_retries} attempts. Continuing anyway...")
        return {"sleep": False, "error": "Max retries exceeded"}

    def wake_up(self, max_retries=10, retry_wake_up_time=1, max_retry_wake_up_time=8):
        """
        Wake up the VLLM server with built-in fault tolerance.
        Automatically retries on failure and logs warnings instead of crashing.
        """
        for attempt in range(max_retries):
            try:
                res = self._make_request("wake_up", method="post", max_retries=2)
                if not self.wait_until_awake(timeout=60.0, poll_interval=1.0):
                    logger.warning(
                        "Wake acknowledged but server did not report steady awake state within timeout"
                    )
                logger.info("VLLM client successfully woken up")
                return res
            except Exception as e:
                logger.warning(f"Wake up attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_wake_up_time)  # Wait before retry
                    retry_wake_up_time = min(retry_wake_up_time * 2, max_retry_wake_up_time)
                    
        logger.error(f"Failed to wake up VLLM client after {max_retries} attempts. Continuing anyway...")
        return {"wake_up": False, "error": "Max retries exceeded"}


# Example usage
if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32)
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(
        "cuda"
    )
    client.update_model_params(model)
