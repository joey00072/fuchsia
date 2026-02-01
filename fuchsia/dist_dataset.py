from fuchsia.vllm_client import VLLMClient
from rich import print
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetClient:
    def __init__(self, client: VLLMClient = None):
        self.client = client if client is not None else VLLMClient()
        logger.info("DatasetClient initialized")
        # self.client.sleep()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        logger.debug(f"Getting item at index {idx}")
        return self.client.get_sample()

    def __iter__(self):
        wait = 1
        while True:
            logger.debug("Attempting to get next sample")
            sample = self.client.get_sample()
            if sample is not None:
                logger.debug("Successfully retrieved sample")
                yield sample
            else:
                logger.warning(f"No sample available, waiting {wait} seconds")
                time.sleep(wait)
                wait *= 2
                continue
            wait = 1


import time

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
