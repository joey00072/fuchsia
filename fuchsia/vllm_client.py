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

# this is so much differ from the original code, but still keeping its spirit

from __future__ import annotations

import atexit
from dataclasses import dataclass
import logging
import time
from typing import Callable, Optional

from peft import PeftModel
import requests
from requests import ConnectionError
import torch
from torch import nn
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


def _as_optional_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_optional_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class _FillSnapshot:
    current_size: int
    is_filling: bool
    fill_started: Optional[float]
    fill_finished: Optional[float]
    heartbeat: Optional[float]
    generated_groups: Optional[int]


@dataclass
class _FillTracker:
    seen_activity: bool = False
    last_heartbeat: Optional[float] = None
    last_fill_started: Optional[float] = None
    last_generated_groups: Optional[int] = None

    def observe(self, snap: _FillSnapshot) -> bool:
        progressed = False
        if snap.heartbeat is not None and (
            self.last_heartbeat is None or snap.heartbeat > self.last_heartbeat
        ):
            progressed = True
        if snap.fill_started is not None and (
            self.last_fill_started is None or snap.fill_started > self.last_fill_started
        ):
            progressed = True
        if snap.generated_groups is not None and (
            self.last_generated_groups is None
            or snap.generated_groups > self.last_generated_groups
        ):
            progressed = True

        self.last_heartbeat = snap.heartbeat
        self.last_fill_started = snap.fill_started
        self.last_generated_groups = snap.generated_groups
        self.seen_activity = True
        return progressed

    def clear_idle_markers(self) -> None:
        self.last_heartbeat = None


class _VLLMTransport:
    """HTTP transport with retry/backoff and small typed defaults."""

    def __init__(self, session: requests.Session, host: str, server_port: int):
        self._session = session
        self.host = host
        self.server_port = server_port
        self.base_url = f"http://{host}:{server_port}"

    def request(
        self,
        endpoint: str,
        *,
        method: str = "get",
        json: Optional[dict] = None,
        timeout: float = 10.0,
        retries: int = 2,
        retry_delay: float = 1.0,
        retry_backoff: float = 1.5,
    ) -> dict:
        url = f"{self.base_url}/{endpoint.strip('/')}/"
        current_delay = max(float(retry_delay), 0.1)
        for attempt in range(retries + 1):
            try:
                response = self._session.request(
                    method=method.upper(),
                    url=url,
                    json=json,
                    timeout=max(float(timeout), 0.1),
                )
                if response.status_code == 200:
                    return response.json() if response.content else {}
                raise RuntimeError(
                    f"HTTP {response.status_code} calling {url}: {response.text}"
                )
            except (requests.exceptions.RequestException, RuntimeError) as exc:
                if attempt >= retries:
                    raise ConnectionError(f"Failed to call {url}: {exc}") from exc
                time.sleep(current_delay)
                current_delay *= max(float(retry_backoff), 1.0)
        raise ConnectionError(f"Unexpected request failure for {url}")

    def request_or_default(
        self,
        endpoint: str,
        *,
        default: dict,
        warn_message: Optional[str] = None,
        warn_level: int = logging.WARNING,
        method: str = "get",
        json: Optional[dict] = None,
        timeout: float = 10.0,
        retries: int = 2,
    ) -> dict:
        try:
            return self.request(
                endpoint,
                method=method,
                json=json,
                timeout=timeout,
                retries=retries,
            )
        except Exception as exc:
            if warn_message:
                logger.log(warn_level, "%s: %s", warn_message, exc)
            return default

    def wait_for_health(self, total_timeout: float, retry_interval: float) -> None:
        if total_timeout <= 0:
            self.request("health", timeout=2.0, retries=0)
            return
        deadline = time.monotonic() + float(total_timeout)
        wait = max(float(retry_interval), 0.1)
        while time.monotonic() < deadline:
            try:
                self.request("health", timeout=2.0, retries=0)
                return
            except Exception:
                time.sleep(wait)
        raise ConnectionError(
            f"The vLLM server can't be reached at {self.host}:{self.server_port} "
            f"after {total_timeout} seconds."
        )


class _CommunicatorSync:
    """Owns NCCL communicator lifecycle and param broadcast/update RPC."""

    def __init__(self, transport: _VLLMTransport, host: str, group_port: int):
        self._transport = transport
        self._host = host
        self._group_port = group_port
        self.rank = 0
        self.pynccl_comm: Optional[PyNcclCommunicator] = None

    def init(self) -> None:
        response = self._transport.request(
            "get_tensor_parallel_size",
            timeout=10.0,
            retries=2,
        )
        tensor_parallel_size = int(response["tensor_parallel_size"])
        world_size = tensor_parallel_size + 1
        self.rank = tensor_parallel_size

        self._transport.request(
            "init_communicator",
            method="post",
            json={
                "host": self._host,
                "port": self._group_port,
                "world_size": world_size,
            },
            timeout=20.0,
            retries=2,
        )

        pg = StatelessProcessGroup.create(
            host=self._host,
            port=self._group_port,
            rank=self.rank,
            world_size=world_size,
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")

    def close(self, *, enabled: bool) -> dict:
        if not enabled:
            return {"close_communicator": True, "skipped": True}
        result = self._transport.request_or_default(
            "close_communicator",
            method="post",
            timeout=10.0,
            retries=2,
            default={"close_communicator": False, "error": "request_failed"},
            warn_message="Failed to close communicator cleanly",
        )
        self.pynccl_comm = None
        return result

    def push_named_param(self, name: str, weights: torch.Tensor) -> None:
        self._transport.request(
            "update_named_param",
            method="post",
            json={
                "name": name,
                "dtype": str(weights.dtype),
                "shape": tuple(weights.shape),
            },
            timeout=15.0,
            retries=2,
        )
        if self.pynccl_comm is None:
            return
        self.pynccl_comm.broadcast(
            weights,
            src=self.rank,
            stream=torch.cuda.current_stream(),
        )
        self.pynccl_comm.group.barrier()


class _ModelSync:
    """Model/LoRA parameter sync logic used by the trainer update step."""

    def __init__(self, communicator: _CommunicatorSync):
        self._communicator = communicator

    def _update_lora_params(self, model: PeftModel) -> None:
        target_modules = model.peft_config["default"].target_modules
        alpha = model.peft_config["default"].lora_alpha
        rank = model.peft_config["default"].r
        weights = {name: param.data for name, param in model.named_parameters()}

        for name, param in model.named_parameters():
            if "lora" in name:
                continue
            if not any(target in name for target in target_modules):
                continue

            if "bias" in name:
                server_name = name.replace("base_model.model.", "")
                self._communicator.push_named_param(server_name, param.data.clone())
                continue

            prefix = name.replace(".base_layer.weight", "")
            a_name = f"{prefix}.lora_A.default.weight"
            b_name = f"{prefix}.lora_B.default.weight"
            delta = (weights[b_name].clone() @ weights[a_name].clone()) * (alpha / rank)
            merged = param.data.clone() + delta
            server_name = (
                name.replace("base_model.model.", "")
                .replace(".base_layer.weight", ".weight")
            )
            self._communicator.push_named_param(server_name, merged)

    def _update_dense_params(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self._communicator.push_named_param(name, param.data)

    def update_model_params(
        self,
        model: nn.Module,
        tokenizer=None,
        *,
        lora: bool = False,
        single_gpu: bool = False,
        lora_path: Optional[str] = None,
    ) -> None:
        if single_gpu:
            if tokenizer is not None and lora_path is not None:
                tokenizer.save_pretrained(lora_path)
            if lora_path is not None:
                model.save_pretrained(lora_path, adapter_name="grpo")
            return

        if lora:
            if not isinstance(model, PeftModel):
                raise ValueError("Model is not a PeftModel")
            self._update_lora_params(model)
            return

        self._update_dense_params(model)


class _BufferController:
    """Rollout queue / sleep-wake orchestration over the server HTTP API."""

    def __init__(self, transport: _VLLMTransport):
        self._transport = transport

    @staticmethod
    def _next_backoff_interval(current: float, *, factor: float, max_interval: float) -> float:
        return min(current * max(float(factor), 1.0), max_interval)

    @staticmethod
    def _is_fill_in_progress(
        status: dict,
        *,
        fill_started: Optional[float] = None,
        fill_finished: Optional[float] = None,
    ) -> bool:
        if fill_started is None:
            fill_started = _as_optional_float(status.get("last_fill_started_at"))
        if fill_finished is None:
            fill_finished = _as_optional_float(status.get("last_fill_finished_at"))
        has_unfinished_fill = (
            fill_started is not None
            and (fill_finished is None or fill_finished < fill_started)
        )
        return bool(
            status.get("is_filling", False)
            or status.get("fill_slot_claimed", False)
            or status.get("generation_in_progress", False)
            or has_unfinished_fill
        )

    def _build_fill_snapshot(self, status: dict) -> _FillSnapshot:
        fill_started = _as_optional_float(status.get("last_fill_started_at"))
        fill_finished = _as_optional_float(status.get("last_fill_finished_at"))
        return _FillSnapshot(
            current_size=int(status.get("current_size", 0)),
            is_filling=self._is_fill_in_progress(
                status,
                fill_started=fill_started,
                fill_finished=fill_finished,
            ),
            fill_started=fill_started,
            fill_finished=fill_finished,
            heartbeat=_as_optional_float(status.get("last_fill_heartbeat_at")),
            generated_groups=_as_optional_int(status.get("last_fill_generated_groups")),
        )

    @staticmethod
    def _fill_completed(snap: _FillSnapshot) -> bool:
        return (
            snap.fill_started is not None
            and snap.fill_finished is not None
            and snap.fill_finished >= snap.fill_started
        )

    def get_sample(self) -> Optional[dict]:
        response = self._transport.request_or_default(
            "get_sample",
            method="post",
            timeout=10.0,
            retries=2,
            default={},
            warn_message="Failed to fetch sample",
        )
        return response.get("sample")

    def empty_buffer(self) -> dict:
        return self._transport.request_or_default(
            "empty_buffer",
            method="post",
            timeout=15.0,
            retries=2,
            default={"empty_buffer": False, "error": "request_failed"},
            warn_message="Failed to empty vLLM buffer",
        )

    def fill_buffer(self, num_samples: Optional[int] = None) -> dict:
        payload = {"num_samples": int(num_samples)} if num_samples is not None else None
        return self._transport.request_or_default(
            "buffer_fill",
            method="post",
            json=payload,
            timeout=30.0,
            retries=2,
            default={"buffer_fill": False, "error": "request_failed"},
            warn_message="Failed to request buffer fill",
        )

    def buffer_status(
        self,
        *,
        timeout: float = 5.0,
        max_retries: int = 0,
        log_failures: bool = False,
    ) -> dict:
        warn_message = (
            "Failed to fetch buffer status" if log_failures else "Buffer status probe failed"
        )
        warn_level = logging.WARNING if log_failures else logging.DEBUG
        return self._transport.request_or_default(
            "buffer_status",
            method="get",
            timeout=timeout,
            retries=max_retries,
            default={},
            warn_message=warn_message,
            warn_level=warn_level,
        )

    def _wait_for_status(
        self,
        predicate: Callable[[dict], bool],
        *,
        timeout: float,
        poll_interval: float,
        max_poll_interval: float,
        backoff_factor: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        current_interval = max(float(poll_interval), 0.1)
        max_interval = max(current_interval, float(max_poll_interval))
        while time.monotonic() < deadline:
            status = self.buffer_status(
                timeout=max(1.0, min(current_interval * 2.0, 8.0)),
                max_retries=0,
                log_failures=False,
            )
            if status and predicate(status):
                return True
            time.sleep(current_interval)
            current_interval = self._next_backoff_interval(
                current_interval,
                factor=backoff_factor,
                max_interval=max_interval,
            )
        return False

    def wait_for_buffer_ready(
        self,
        *,
        min_size: int = 1,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
        filling_grace_timeout: float = 600.0,
        assume_filling: bool = False,
    ) -> bool:
        start_time = time.monotonic()
        base_deadline = start_time + timeout
        filling_deadline: Optional[float] = (
            start_time + float(filling_grace_timeout) if assume_filling else None
        )
        current_interval = max(float(poll_interval), 0.1)
        max_interval = max(current_interval, float(max_poll_interval))
        tracker = _FillTracker(seen_activity=bool(assume_filling))

        while True:
            now = time.monotonic()
            status = self.buffer_status(
                timeout=max(3.0, min(current_interval * 2.0, 8.0)),
                max_retries=0,
                log_failures=False,
            )
            if status:
                snap = self._build_fill_snapshot(status)
                if snap.current_size >= min_size and not snap.is_filling:
                    return True
                if snap.is_filling:
                    progressed = tracker.observe(snap)
                    if filling_deadline is None or progressed:
                        filling_deadline = now + float(filling_grace_timeout)
                else:
                    if filling_deadline is not None and self._fill_completed(snap):
                        return False
                    if not tracker.seen_activity:
                        filling_deadline = None
                        tracker.clear_idle_markers()

            if filling_deadline is not None and now >= filling_deadline:
                return False
            if filling_deadline is None and now >= base_deadline:
                return False

            time.sleep(current_interval)
            current_interval = self._next_backoff_interval(
                current_interval,
                factor=backoff_factor,
                max_interval=max_interval,
            )

    def wait_until_sleeping(
        self,
        *,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
    ) -> bool:
        def _sleeping(status: dict) -> bool:
            return bool(status.get("is_sleeping", False)) and not bool(
                status.get("sleep_requested", False)
            ) and not bool(status.get("is_filling", False))

        return self._wait_for_status(
            _sleeping,
            timeout=timeout,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            backoff_factor=backoff_factor,
        )

    def wait_until_awake(
        self,
        *,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
    ) -> bool:
        def _awake(status: dict) -> bool:
            return (not bool(status.get("is_sleeping", False))) and (
                not bool(status.get("sleep_requested", False))
            )

        return self._wait_for_status(
            _awake,
            timeout=timeout,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            backoff_factor=backoff_factor,
        )

    def _post_control_command(
        self,
        *,
        endpoint: str,
        success_key: str,
        wait_fn: Callable[..., bool],
        wait_timeout: float,
        max_retries: int,
        initial_delay: float,
        max_delay: float,
    ) -> dict:
        delay = max(float(initial_delay), 0.1)
        capped_delay = max(delay, float(max_delay))
        for attempt in range(max_retries):
            try:
                response = self._transport.request(
                    endpoint,
                    method="post",
                    timeout=20.0,
                    retries=2,
                )
                # Backward compatibility: older wake endpoints may omit "wake_up".
                if (
                    endpoint == "wake_up"
                    and success_key not in response
                    and "error" not in response
                ):
                    response[success_key] = True
                    response["inferred_success"] = True
                if response.get(success_key, False):
                    if not wait_fn(timeout=wait_timeout, poll_interval=1.0):
                        logger.warning(
                            "%s acknowledged but server did not reach steady state",
                            endpoint,
                        )
                    return response
            except Exception as exc:
                logger.warning("%s attempt %s failed: %s", endpoint, attempt + 1, exc)

            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2.0, capped_delay)
        return {success_key: False, "error": "Max retries exceeded"}

    def sleep(
        self,
        *,
        max_retries: int = 100,
        retry_sleep_time: float = 2.0,
        max_retry_sleep_time: float = 8.0,
    ) -> dict:
        return self._post_control_command(
            endpoint="sleep",
            success_key="sleep",
            wait_fn=self.wait_until_sleeping,
            wait_timeout=60.0,
            max_retries=max_retries,
            initial_delay=retry_sleep_time,
            max_delay=max_retry_sleep_time,
        )

    def wake_up(
        self,
        *,
        max_retries: int = 10,
        retry_wake_up_time: float = 1.0,
        max_retry_wake_up_time: float = 8.0,
    ) -> dict:
        return self._post_control_command(
            endpoint="wake_up",
            success_key="wake_up",
            wait_fn=self.wait_until_awake,
            wait_timeout=60.0,
            max_retries=max_retries,
            initial_delay=retry_wake_up_time,
            max_delay=max_retry_wake_up_time,
        )


class VLLMClient:
    """Facade preserving existing public API while delegating responsibilities."""

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
        self._init_communicator = bool(init_communicator)
        self._connection_timeout = float(connection_timeout)

        self._transport = _VLLMTransport(self.session, host, server_port)
        self._buffer = _BufferController(self._transport)
        self._communicator = _CommunicatorSync(self._transport, host, group_port)
        self._model_sync = _ModelSync(self._communicator)

        self._connect_with_retries()
        if self._init_communicator:
            self._init_communicator_with_retries()
            atexit.register(self.close_communicator)

    @property
    def rank(self) -> int:
        return self._communicator.rank

    @property
    def pynccl_comm(self) -> Optional[PyNcclCommunicator]:
        return self._communicator.pynccl_comm

    def check_server(self, total_timeout: float = 180.0, retry_interval: float = 2.0) -> None:
        self._transport.wait_for_health(total_timeout, retry_interval)

    def _connect_with_retries(self) -> None:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.check_server(self._connection_timeout)
                return
            except Exception as exc:
                logger.warning("Connection attempt %s failed: %s", attempt + 1, exc)
                if attempt < max_attempts - 1:
                    time.sleep(5.0)
        logger.error(
            "Failed to connect to VLLM server; subsequent client operations may fail."
        )

    def _init_communicator_with_retries(self) -> None:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.init_communicator()
                return
            except Exception as exc:
                logger.warning(
                    "Communicator init attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
                if attempt < max_attempts - 1:
                    time.sleep(3.0)
        logger.error("Failed to initialize communicator; running without NCCL sync.")
        self._communicator.rank = 0
        self._communicator.pynccl_comm = None

    def init_communicator(self) -> None:
        self._communicator.init()

    def close_communicator(self) -> dict:
        return self._communicator.close(enabled=self._init_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self._communicator.push_named_param(name, weights)

    def update_model_params(
        self,
        model: nn.Module,
        tokenizer=None,
        lora: bool = False,
        single_gpu: bool = False,
        lora_path: Optional[str] = None,
    ) -> None:
        self._model_sync.update_model_params(
            model,
            tokenizer,
            lora=lora,
            single_gpu=single_gpu,
            lora_path=lora_path,
        )

    def get_sample(self) -> Optional[dict]:
        return self._buffer.get_sample()

    def empty_buffer(self) -> dict:
        return self._buffer.empty_buffer()

    def fill_buffer(self, num_samples: Optional[int] = None) -> dict:
        return self._buffer.fill_buffer(num_samples)

    def buffer_status(
        self,
        *,
        timeout: float = 5.0,
        max_retries: int = 0,
        log_failures: bool = False,
    ) -> dict:
        return self._buffer.buffer_status(
            timeout=timeout,
            max_retries=max_retries,
            log_failures=log_failures,
        )

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
        return self._buffer.wait_for_buffer_ready(
            min_size=min_size,
            timeout=timeout,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            backoff_factor=backoff_factor,
            filling_grace_timeout=filling_grace_timeout,
            assume_filling=assume_filling,
        )

    def wait_until_sleeping(
        self,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        max_poll_interval: float = 5.0,
        backoff_factor: float = 1.5,
    ) -> bool:
        return self._buffer.wait_until_sleeping(
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
        return self._buffer.wait_until_awake(
            timeout=timeout,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            backoff_factor=backoff_factor,
        )

    def sleep(
        self,
        max_retries: int = 100,
        retry_sleep_time: float = 2.0,
        max_retry_sleep_time: float = 8.0,
    ) -> dict:
        return self._buffer.sleep(
            max_retries=max_retries,
            retry_sleep_time=retry_sleep_time,
            max_retry_sleep_time=max_retry_sleep_time,
        )

    def wake_up(
        self,
        max_retries: int = 10,
        retry_wake_up_time: float = 1.0,
        max_retry_wake_up_time: float = 8.0,
    ) -> dict:
        return self._buffer.wake_up(
            max_retries=max_retries,
            retry_wake_up_time=retry_wake_up_time,
            max_retry_wake_up_time=max_retry_wake_up_time,
        )
