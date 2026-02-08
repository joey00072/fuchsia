import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

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


class DatasetClient:
    def __init__(
        self,
        client: Optional[VLLMClient] = None,
        transfer_mode: Optional[str] = None,
        queue_dir: Optional[str] = None,
        poll_interval: float = 1.0,
    ):
        self.transfer_mode = normalize_rollout_transfer_mode(
            transfer_mode or os.getenv("FUCHSIA_SAMPLE_TRANSFER_MODE", "api")
        )

        self.poll_interval = float(
            os.getenv(
                "FUCHSIA_SAMPLE_TRANSFER_POLL_INTERVAL",
                str(poll_interval),
            )
        )

        if self.transfer_mode == "api":
            self.client = client or VLLMClient()
            self.receiver: BaseRolloutReceiver = APIRolloutReceiver(self.client)
            logger.info("DatasetClient initialized with API transfer")
        else:
            resolved_queue_dir = (
                queue_dir
                or os.getenv("FUCHSIA_SAMPLE_TRANSFER_DIR", "/tmp/fuchsia_sample_queue")
            )
            self.receiver = FileSystemRolloutReceiver(resolved_queue_dir)
            logger.info(
                "DatasetClient initialized with filesystem transfer (%s)",
                self.receiver.queue.queue_dir,
            )

    def __len__(self):
        # Streaming dataset client does not have a finite local size.
        return 0

    def __getitem__(self, idx):
        logger.debug(f"Getting item at index {idx}")
        return self._get_sample()

    def _get_sample(self):
        return self.receiver.get()

    def __iter__(self):
        wait = self.poll_interval
        while True:
            logger.debug("Attempting to get next sample")
            sample = self._get_sample()
            if sample is not None:
                logger.debug("Successfully retrieved sample")
                yield sample
            else:
                logger.warning(f"No sample available, waiting {wait} seconds")
                time.sleep(wait)
                wait = min(wait * 2, 8.0)
                continue
            wait = self.poll_interval

if __name__ == "__main__":
    logger.info("Starting main execution")
    client = VLLMClient()
    dataset = DatasetClient(client)
    num_samples = 10
    logger.info(f"Fetching {num_samples} samples")
    for idx, item in enumerate(dataset):
        logger.info(f"Processing sample {idx}")
        print(item["completions"])
        time.sleep(0.2)
