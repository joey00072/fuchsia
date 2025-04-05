from fuchsia.dist_dataset import DatasetClient

from rich import print
import time
import torch
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from mock_server import MockClient
client = MockClient()
dist_dataset = DatasetClient(client)


file_name = "train_data_16k.jsonl"

with open(file_name, "w") as f: 
    for idx, item in enumerate(dist_dataset):
        print(item)
        f.write(json.dumps(item))
        f.write("\n")
        print(idx)
        time.sleep(4)
        if idx > 16:
            break
    # print(item["completions"])
    # print(item["item"][0]["text"])
    # rewards = torch.tensor(item["rewards"])
    # print(idx,rewards)
    # time.sleep(20)
    # if idx > 10:
    #     break







