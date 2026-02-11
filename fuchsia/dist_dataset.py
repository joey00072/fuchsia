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
        max_seq_len: Optional[int] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        debug: bool = False,
        non_blocking: bool = False,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if max_seq_len is not None and max_seq_len <= 1:
            raise ValueError(f"max_seq_len must be > 1, got {max_seq_len}")

        self.source = source
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else None
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

    def _build_batch_from_token_ids(
        self,
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
        completion_logprobs: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
        pad_token_id = self._get_pad_token_id()
        rows: list[tuple[list[int], list[int], list[float], list[int]]] = []
        for prompt_token_ids, completion_token_ids, token_logprobs in zip(
            prompt_ids, completion_ids, completion_logprobs
        ):
            prompt_row = list(prompt_token_ids)
            completion_row = list(completion_token_ids)
            logprob_row = list(token_logprobs)

            prompt_len = len(prompt_row)
            completion_len = len(completion_row)
            if prompt_len <= 0:
                raise RuntimeError("Malformed rollout payload: prompt_ids must be non-empty")
            if completion_len <= 0:
                raise RuntimeError("Malformed rollout payload: completion_ids must be non-empty")
            if completion_len != len(logprob_row):
                raise RuntimeError(
                    "Malformed rollout payload: completion_ids and completion_logprobs length mismatch "
                    f"({completion_len} != {len(logprob_row)})"
                )

            if self.max_seq_len is not None:
                total_len = prompt_len + completion_len
                if total_len > self.max_seq_len:
                    max_completion_len = self.max_seq_len - prompt_len
                    if max_completion_len <= 0:
                        # Keep prompt tail and reserve one token for completion loss.
                        keep_prompt_len = max(1, self.max_seq_len - 1)
                        prompt_row = prompt_row[-keep_prompt_len:]
                        prompt_len = len(prompt_row)
                        max_completion_len = self.max_seq_len - prompt_len
                    if max_completion_len <= 0:
                        raise RuntimeError(
                            "Malformed rollout payload: max_seq_len too small for prompt/completion training"
                        )
                    completion_row = completion_row[:max_completion_len]
                    logprob_row = logprob_row[:max_completion_len]
                    completion_len = len(completion_row)
                    if completion_len <= 0:
                        raise RuntimeError(
                            "Malformed rollout payload: completion truncated to zero length"
                        )

            seq = prompt_row + completion_row
            rows.append((prompt_row, completion_row, logprob_row, seq))

        if not rows:
            raise RuntimeError("Cannot build tensor batch from empty rollout payload")

        batch_max_seq_len = max(len(seq) for _, _, _, seq in rows)
        if batch_max_seq_len <= 1:
            raise RuntimeError(
                f"Malformed rollout payload: max sequence length must be > 1, got {batch_max_seq_len}"
            )

        current_batch_size = len(rows)
        output_ids = torch.full((current_batch_size, batch_max_seq_len), pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((current_batch_size, batch_max_seq_len), dtype=torch.bool)
        old_policy_log_probs = torch.zeros((current_batch_size, batch_max_seq_len - 1), dtype=torch.float32)

        decoded_prompt_ids: list[list[int]] = []
        for row_idx, (prompt_token_ids, completion_token_ids, token_logprobs, seq) in enumerate(rows):
            prompt_len = len(prompt_token_ids)
            completion_len = len(completion_token_ids)
            seq_len = len(seq)

            output_ids[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            loss_mask[row_idx, prompt_len : prompt_len + completion_len] = True
            start_idx = prompt_len - 1
            old_policy_log_probs[row_idx, start_idx : start_idx + completion_len] = torch.tensor(
                token_logprobs, dtype=torch.float32
            )
            decoded_prompt_ids.append(prompt_token_ids)

        decoded_prompts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in decoded_prompt_ids]
        decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        return output_ids, loss_mask[:, 1:], old_policy_log_probs, decoded_prompts, decoded_outputs

    def _prepare_next_batch(self, source_iter: Iterator[dict[str, Any]]) -> dict[str, Any]:
        prompt_ids: list[list[int]] = []
        completions: list[str] = []
        completion_ids: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        rewards: list[list[float]] = []
        mean_rewards: list[float] = []
        std_rewards: list[float] = []
        ignore_sample: list[bool] = []
        group_size: Optional[int] = None

        for _ in range(self.batch_size):
            item = next(source_iter)
            if not isinstance(item, dict):
                raise RuntimeError(f"Malformed rollout payload: expected dict, got {type(item)}")

            item_completions = item.get("completions")
            item_rewards = item.get("rewards")
            item_prompt_ids = item.get("prompt_ids")
            item_completion_ids = item.get("completion_ids")
            item_completion_logprobs = item.get("completion_logprobs")

            if not isinstance(item_completions, list) or not item_completions:
                raise RuntimeError("Malformed rollout payload: completions must be a non-empty list")
            if not isinstance(item_rewards, list) or len(item_rewards) != len(item_completions):
                raise RuntimeError("Malformed rollout payload: rewards must be a list matching completions length")
            if not isinstance(item_prompt_ids, list) or not all(isinstance(v, int) for v in item_prompt_ids):
                raise RuntimeError("Malformed rollout payload: prompt_ids must be a non-empty list[int]")
            if len(item_prompt_ids) == 0:
                raise RuntimeError("Malformed rollout payload: prompt_ids cannot be empty")
            if not isinstance(item_completion_ids, list) or len(item_completion_ids) != len(item_completions):
                raise RuntimeError("Malformed rollout payload: completion_ids must match completions length")
            if not isinstance(item_completion_logprobs, list) or len(item_completion_logprobs) != len(item_completions):
                raise RuntimeError("Malformed rollout payload: completion_logprobs must match completions length")

            item_group_size = len(item_completions)
            if group_size is None:
                group_size = item_group_size
            elif item_group_size != group_size:
                raise RuntimeError(
                    f"Malformed rollout payload: inconsistent group_size ({item_group_size} != {group_size})"
                )

            rewards.append([float(reward) for reward in item_rewards])
            mean_rewards.append(float(item.get("mean", 0.0)))
            std_rewards.append(float(item.get("std", 0.0)))

            finish_reasons = item.get("finish_reason")
            for idx in range(item_group_size):
                completion_text = str(item_completions[idx])
                completion_token_ids = item_completion_ids[idx]
                completion_token_logprobs = item_completion_logprobs[idx]

                if not isinstance(completion_token_ids, list) or not all(
                    isinstance(token_id, int) for token_id in completion_token_ids
                ):
                    raise RuntimeError("Malformed rollout payload: each completion_ids entry must be list[int]")
                if not isinstance(completion_token_logprobs, list):
                    raise RuntimeError("Malformed rollout payload: each completion_logprobs entry must be list[float]")

                completion_ids.append([int(token_id) for token_id in completion_token_ids])
                completion_logprobs.append([float(logprob) for logprob in completion_token_logprobs])
                prompt_ids.append([int(token_id) for token_id in item_prompt_ids])
                completions.append(completion_text)

                finish_reason = None
                if isinstance(finish_reasons, list) and idx < len(finish_reasons):
                    finish_reason = finish_reasons[idx]
                ignore_sample.append(finish_reason != "length")

        if group_size is None:
            raise RuntimeError("Malformed rollout payload: empty trainer batch")

        outputs, shifted_loss_mask, old_policy_log_probs, decoded_prompts, decoded_outputs = (
            self._build_batch_from_token_ids(prompt_ids, completion_ids, completion_logprobs)
        )

        avg_token_lengths = 0.0
        if completion_ids:
            avg_token_lengths = sum(len(ids) for ids in completion_ids) / len(completion_ids)

        return {
            "outputs": outputs.to(self.device, non_blocking=self.non_blocking),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "mean_rewards": torch.tensor(mean_rewards, dtype=torch.float32, device=self.device),
            "std_rewards": torch.tensor(std_rewards, dtype=torch.float32, device=self.device),
            "loss_mask": shifted_loss_mask.to(self.device, non_blocking=self.non_blocking),
            "ignore_sample": torch.tensor(ignore_sample, dtype=torch.float32, device=self.device),
            "server_info": {
                "group_size": group_size,
                "old_policy_log_probs": old_policy_log_probs.to(self.device, non_blocking=self.non_blocking),
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
