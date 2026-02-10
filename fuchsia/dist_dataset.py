from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from fuchsia.rollout_queue import FileSystemRolloutQueue, normalize_rollout_transfer_mode
from fuchsia.vllm_client import VLLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseRolloutReceiver(ABC):
    mode: str = ""

    @abstractmethod
    def get(self) -> Optional[dict]:
        raise NotImplementedError


class APIRolloutReceiver(BaseRolloutReceiver):
    mode = "api"

    def __init__(self, client: VLLMClient):
        self.client = client

    def get(self) -> Optional[dict]:
        return self.client.get_sample()


class FileSystemRolloutReceiver(BaseRolloutReceiver):
    mode = "filesystem"

    def __init__(self, queue_dir: str):
        self.queue = FileSystemRolloutQueue(queue_dir)

    def get(self) -> Optional[dict]:
        return self.queue.get()


def _resolve_distributed_context(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> tuple[int, int]:
    if rank is not None and world_size is not None:
        return int(rank), max(int(world_size), 1)

    if dist.is_available() and dist.is_initialized():
        detected_rank = dist.get_rank()
        detected_world_size = dist.get_world_size()
        return detected_rank, max(detected_world_size, 1)

    env_rank = os.getenv("RANK", "0")
    env_world_size = os.getenv("WORLD_SIZE", "1")
    try:
        detected_rank = int(env_rank)
    except ValueError:
        detected_rank = 0
    try:
        detected_world_size = int(env_world_size)
    except ValueError:
        detected_world_size = 1
    return detected_rank, max(detected_world_size, 1)


class RolloutStreamDataset(IterableDataset):
    """Streaming rollout dataset backed by API or filesystem queue."""

    def __init__(
        self,
        client: Optional[VLLMClient] = None,
        transfer_mode: Optional[str] = None,
        queue_dir: Optional[str] = None,
        poll_interval: float = 1.0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        super().__init__()
        self.transfer_mode = normalize_rollout_transfer_mode(
            transfer_mode or os.getenv("FUCHSIA_SAMPLE_TRANSFER_MODE", "api")
        )
        self.poll_interval = float(
            os.getenv(
                "FUCHSIA_SAMPLE_TRANSFER_POLL_INTERVAL",
                str(poll_interval),
            )
        )
        self.rank, self.world_size = _resolve_distributed_context(rank, world_size)

        if self.transfer_mode == "api":
            self.client = client or VLLMClient()
            self.receiver: BaseRolloutReceiver = APIRolloutReceiver(self.client)
            logger.info(
                "RolloutStreamDataset initialized with API transfer (rank=%s world_size=%s)",
                self.rank,
                self.world_size,
            )
        else:
            resolved_queue_dir = (
                queue_dir
                or os.getenv("FUCHSIA_SAMPLE_TRANSFER_DIR", "/tmp/fuchsia_sample_queue")
            )
            self.receiver = FileSystemRolloutReceiver(resolved_queue_dir)
            logger.info(
                "RolloutStreamDataset initialized with filesystem transfer (%s, rank=%s world_size=%s)",
                self.receiver.queue.queue_dir,
                self.rank,
                self.world_size,
            )

    def __len__(self) -> int:
        # Streaming rollout source does not expose a finite local size.
        return 0

    def __getitem__(self, idx: int) -> Optional[dict]:
        logger.debug("Getting item at index %s", idx)
        return self._get_sample()

    def _get_sample(self) -> Optional[dict]:
        return self.receiver.get()

    def _sample_matches_rank(self, sample: dict[str, Any]) -> bool:
        target_rank = sample.get("target_rank")
        if target_rank is None:
            return True
        try:
            parsed_rank = int(target_rank)
        except (TypeError, ValueError):
            return True
        if parsed_rank < 0:
            return True
        return parsed_rank == self.rank

    def __iter__(self) -> Iterator[dict[str, Any]]:
        wait = self.poll_interval
        while True:
            sample = self._get_sample()
            if sample is None:
                logger.debug("No sample available on rank %s, waiting %.2fs", self.rank, wait)
                time.sleep(wait)
                wait = min(wait * 2, 8.0)
                continue
            if not isinstance(sample, dict):
                logger.warning("Ignoring malformed rollout sample of type %s", type(sample))
                continue
            if not self._sample_matches_rank(sample):
                logger.debug("Skipping sample for rank %s", self.rank)
                wait = self.poll_interval
                continue

            wait = self.poll_interval
            sample["_trainer_rank"] = self.rank
            sample["_trainer_world_size"] = self.world_size
            yield sample


class PreparedRolloutBatchDataset(IterableDataset):
    """Trainer-facing dataset that fetches, prepares, and places tensors on device."""

    def __init__(
        self,
        source: IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        debug: bool = False,
        non_blocking: bool = False,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.source = source
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.debug = debug
        self.non_blocking = non_blocking
        self.rank, self.world_size = _resolve_distributed_context()

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        source_iter = iter(self.source)
        while True:
            yield self._prepare_next_batch(source_iter)

    def _get_pad_token_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return 0

    def _build_batch_from_server_tokens(
        self,
        prompts: list[str],
        completion_ids: list[list[int]],
        completion_logprobs: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
        pad_token_id = self._get_pad_token_id()
        encoded_prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_input_ids = encoded_prompts["input_ids"]
        prompt_attention_mask = encoded_prompts["attention_mask"]
        decoded_prompts = self.tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=False)

        prompt_token_lists = [
            prompt_input_ids[idx][prompt_attention_mask[idx].bool()].tolist()
            for idx in range(prompt_input_ids.shape[0])
        ]
        sequences = [
            prompt_tokens + completion_token_ids
            for prompt_tokens, completion_token_ids in zip(prompt_token_lists, completion_ids)
        ]
        max_seq_len = max(len(seq) for seq in sequences)
        current_batch_size = len(sequences)

        output_ids = torch.full((current_batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((current_batch_size, max_seq_len), dtype=torch.bool)
        old_policy_log_probs = torch.zeros((current_batch_size, max_seq_len - 1), dtype=torch.float32)

        for row_idx, (prompt_tokens, completion_token_ids, completion_token_logprobs, seq) in enumerate(
            zip(prompt_token_lists, completion_ids, completion_logprobs, sequences)
        ):
            seq_len = len(seq)
            prompt_len = len(prompt_tokens)
            completion_len = len(completion_token_ids)

            if seq_len > 0:
                output_ids[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            if completion_len > 0:
                loss_mask[row_idx, prompt_len : prompt_len + completion_len] = True
                start_idx = max(prompt_len - 1, 0)
                old_policy_log_probs[row_idx, start_idx : start_idx + completion_len] = torch.tensor(
                    completion_token_logprobs, dtype=torch.float32
                )

        decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        return output_ids, loss_mask[:, 1:], old_policy_log_probs, decoded_prompts, decoded_outputs

    def _build_batch_from_server_completion_ids(
        self,
        prompts: list[str],
        completion_ids: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
        pad_token_id = self._get_pad_token_id()
        encoded_prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_input_ids = encoded_prompts["input_ids"]
        prompt_attention_mask = encoded_prompts["attention_mask"]
        decoded_prompts = self.tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=False)

        prompt_token_lists = [
            prompt_input_ids[idx][prompt_attention_mask[idx].bool()].tolist()
            for idx in range(prompt_input_ids.shape[0])
        ]
        sequences = [
            prompt_tokens + completion_token_ids
            for prompt_tokens, completion_token_ids in zip(prompt_token_lists, completion_ids)
        ]
        max_seq_len = max(len(seq) for seq in sequences)
        current_batch_size = len(sequences)

        output_ids = torch.full((current_batch_size, max_seq_len), pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((current_batch_size, max_seq_len), dtype=torch.bool)

        for row_idx, (prompt_tokens, completion_token_ids, seq) in enumerate(
            zip(prompt_token_lists, completion_ids, sequences)
        ):
            seq_len = len(seq)
            prompt_len = len(prompt_tokens)
            completion_len = len(completion_token_ids)
            if seq_len > 0:
                output_ids[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            if completion_len > 0:
                loss_mask[row_idx, prompt_len : prompt_len + completion_len] = True

        decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        return output_ids, loss_mask[:, 1:], decoded_prompts, decoded_outputs

    def _can_use_server_logprobs(
        self,
        completions: list[str],
        completion_ids: list[list[int]],
        completion_logprobs: list[list[float] | None],
    ) -> bool:
        for completion_text, completion_token_ids, completion_token_logprobs in zip(
            completions, completion_ids, completion_logprobs
        ):
            if not isinstance(completion_token_ids, list):
                return False
            if not isinstance(completion_token_logprobs, list):
                return False
            if len(completion_token_ids) != len(completion_token_logprobs):
                return False
            text_token_ids = self.tokenizer(completion_text, add_special_tokens=False)["input_ids"]
            if text_token_ids != completion_token_ids:
                return False
        return True

    def _prepare_next_batch(self, source_iter: Iterator[dict[str, Any]]) -> dict[str, Any]:
        while True:
            inputs_texts: list[str] = []
            completions: list[str] = []
            completion_ids: list[list[int]] = []
            completion_logprobs: list[list[float] | None] = []
            rewards: list[list[float]] = []
            mean_rewards: list[float] = []
            std_rewards: list[float] = []
            ignore_sample: list[bool] = []
            group_size: Optional[int] = None
            malformed_batch = False

            for _ in range(self.batch_size):
                item = next(source_iter)
                if not isinstance(item, dict):
                    malformed_batch = True
                    break

                item_completions = item.get("completions")
                item_rewards = item.get("rewards")
                if not isinstance(item_completions, list) or not item_completions:
                    malformed_batch = True
                    break
                if not isinstance(item_rewards, list) or len(item_rewards) != len(item_completions):
                    malformed_batch = True
                    break

                item_group_size = len(item_completions)
                if group_size is None:
                    group_size = item_group_size
                elif item_group_size != group_size:
                    malformed_batch = True
                    break

                rewards.append(item_rewards)
                mean_rewards.append(float(item.get("mean", 0.0)))
                std_rewards.append(float(item.get("std", 0.0)))

                item_completion_ids = item.get("completion_ids")
                item_completion_logprobs = item.get("completion_logprobs")
                finish_reasons = item.get("finish_reason")
                prompt_text = str(item.get("inputs", ""))

                for idx in range(item_group_size):
                    completion_text = str(item_completions[idx])
                    inputs_texts.append(prompt_text)
                    completions.append(completion_text)

                    completion_token_ids: list[int]
                    if isinstance(item_completion_ids, list) and idx < len(item_completion_ids):
                        candidate_ids = item_completion_ids[idx]
                        if isinstance(candidate_ids, list):
                            completion_token_ids = [int(token_id) for token_id in candidate_ids]
                        else:
                            completion_token_ids = self.tokenizer(
                                completion_text, add_special_tokens=False
                            )["input_ids"]
                    else:
                        completion_token_ids = self.tokenizer(
                            completion_text, add_special_tokens=False
                        )["input_ids"]
                    completion_ids.append(completion_token_ids)

                    sample_completion_logprobs: list[float] | None = None
                    if isinstance(item_completion_logprobs, list) and idx < len(item_completion_logprobs):
                        candidate_logprobs = item_completion_logprobs[idx]
                        if isinstance(candidate_logprobs, list):
                            sample_completion_logprobs = [
                                float(logprob) for logprob in candidate_logprobs
                            ]
                    completion_logprobs.append(sample_completion_logprobs)

                    finish_reason = None
                    if isinstance(finish_reasons, list) and idx < len(finish_reasons):
                        finish_reason = finish_reasons[idx]
                    ignore_sample.append(finish_reason != "length")

            if malformed_batch or group_size is None:
                logger.warning(
                    "Skipping malformed rollout batch on rank %s while preparing dataset batch",
                    self.rank,
                )
                continue

            if any(len(group_rewards) != group_size for group_rewards in rewards):
                continue

            use_server_logprobs = self._can_use_server_logprobs(
                completions, completion_ids, completion_logprobs
            )
            old_policy_log_probs = None
            if use_server_logprobs:
                typed_completion_logprobs = [
                    values if isinstance(values, list) else []
                    for values in completion_logprobs
                ]
                outputs, shifted_loss_mask, old_policy_log_probs, decoded_prompts, decoded_outputs = (
                    self._build_batch_from_server_tokens(
                        inputs_texts, completion_ids, typed_completion_logprobs
                    )
                )
            else:
                outputs, shifted_loss_mask, decoded_prompts, decoded_outputs = (
                    self._build_batch_from_server_completion_ids(inputs_texts, completion_ids)
                )

            avg_token_lengths = 0.0
            if completions:
                total = 0
                for completion in completions:
                    total += len(self.tokenizer.encode(completion, add_special_tokens=False))
                avg_token_lengths = total / len(completions)

            return {
                "outputs": outputs.to(self.device, non_blocking=self.non_blocking),
                "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
                "mean_rewards": torch.tensor(mean_rewards, dtype=torch.float32, device=self.device),
                "std_rewards": torch.tensor(std_rewards, dtype=torch.float32, device=self.device),
                "loss_mask": shifted_loss_mask.to(self.device, non_blocking=self.non_blocking),
                "ignore_sample": torch.tensor(ignore_sample, dtype=torch.int8, device=self.device),
                "server_info": {
                    "group_size": group_size,
                    "old_policy_log_probs": (
                        None
                        if old_policy_log_probs is None
                        else old_policy_log_probs.to(self.device, non_blocking=self.non_blocking)
                    ),
                },
                "metrics": {
                    "prompt": decoded_prompts,
                    "completions": completions,
                    "avg_token_lengths": avg_token_lengths,
                    "decoded_outputs": decoded_outputs,
                },
            }


class DatasetClient(RolloutStreamDataset):
    """Backward-compatible alias for existing trainer bootstrap code."""
