from fuchsia.dist_dataset import DatasetClient
from rich import print
import time
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dist_dataset = DatasetClient()



for idx, item in enumerate(dist_dataset):
    print(item["completions"])
    print(item["item"][0]["text"])
    rewards = torch.tensor(item["rewards"])
    print(idx,rewards)
    time.sleep(20)
    if idx > 10:
        break







