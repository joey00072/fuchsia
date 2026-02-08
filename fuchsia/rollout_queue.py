import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_rollout_transfer_mode(mode: Optional[str]) -> str:
    normalized = (mode or "api").lower()
    aliases = {
        "api": "api",
        "http": "api",
        "filesystem": "filesystem",
        "fs": "filesystem",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported rollout transfer mode: {mode}. "
            "Supported values are: ['api', 'filesystem'] (http is accepted as alias for api)."
        )
    return aliases[normalized]


class BaseRolloutQueue(ABC):
    mode: str = ""
    queue_dir: Optional[Path] = None

    @abstractmethod
    def put(self, rollout: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def put_many(self, rollouts: list[Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def qsize(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> int:
        raise NotImplementedError


class InMemoryRolloutQueue(BaseRolloutQueue):
    mode = "api"

    def __init__(self):
        self._buffer: deque[Any] = deque()

    def put(self, rollout: Any) -> None:
        self._buffer.append(rollout)

    def put_many(self, rollouts: list[Any]) -> None:
        self._buffer.extend(rollouts)

    def get(self) -> Optional[Any]:
        if not self._buffer:
            return None
        return self._buffer.popleft()

    def qsize(self) -> int:
        return len(self._buffer)

    def clear(self) -> int:
        removed = len(self._buffer)
        self._buffer.clear()
        return removed


class FileSystemRolloutQueue(BaseRolloutQueue):
    mode = "filesystem"

    def __init__(self, queue_dir: str | Path):
        self.queue_dir = Path(queue_dir).expanduser()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._seq_lock = threading.Lock()

    def _next_message_name(self) -> str:
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        ts_ns = time.time_ns()
        return f"{ts_ns:020d}_{seq:08d}_{uuid.uuid4().hex}.rollout.json"

    def put(self, rollout: Any) -> None:
        tmp_path = self.queue_dir / f".tmp_{os.getpid()}_{uuid.uuid4().hex}.json"
        msg_path = self.queue_dir / self._next_message_name()
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(rollout, handle, ensure_ascii=False)
        os.replace(tmp_path, msg_path)

    def put_many(self, rollouts: list[Any]) -> None:
        for rollout in rollouts:
            self.put(rollout)

    def _iter_message_paths(self):
        patterns = ("*.rollout.json", "*.msg.json")
        paths = []
        for pattern in patterns:
            paths.extend(self.queue_dir.glob(pattern))
        yield from sorted(paths)

    def get(self) -> Optional[Any]:
        pid = os.getpid()
        for msg_path in self._iter_message_paths():
            claim_path = msg_path.with_name(
                f"{msg_path.name}.claim.{pid}.{uuid.uuid4().hex}"
            )
            try:
                os.replace(msg_path, claim_path)
            except FileNotFoundError:
                continue
            except OSError:
                continue

            try:
                with claim_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception as exc:
                logger.warning("Failed to read rollout file %s: %s", claim_path, exc)
                return None
            finally:
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
        return None

    def qsize(self) -> int:
        return sum(1 for _ in self._iter_message_paths())

    def clear(self) -> int:
        removed = 0
        for pattern in (
            "*.rollout.json",
            "*.rollout.json.claim.*",
            "*.msg.json",
            "*.msg.json.claim.*",
            ".tmp_*.json",
        ):
            for path in self.queue_dir.glob(pattern):
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning("Failed to remove queue file %s: %s", path, exc)
        return removed


def create_rollout_queue(
    mode: Optional[str],
    queue_dir: str | Path,
    clear_on_start: bool = False,
) -> BaseRolloutQueue:
    normalized_mode = normalize_rollout_transfer_mode(mode)
    if normalized_mode == "api":
        return InMemoryRolloutQueue()

    queue = FileSystemRolloutQueue(queue_dir=queue_dir)
    if clear_on_start:
        queue.clear()
    return queue
