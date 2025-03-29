from fuchsia.vllm_client import VLLMClient, VLLMClientConfig

class DatasetClient:
    def __init__(self, config: VLLMClientConfig):
        self.client = VLLMClient(config)
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.client.get_sample()