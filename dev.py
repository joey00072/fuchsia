import torch

x = torch.rand(2,2)

x = x.to(torch.device("cuda"))
print(x)
