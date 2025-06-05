import socket
import psutil
import torch
import os
from typing import Any

def get_ip_addresses():
    ip_info = []

    hostname = socket.gethostname()
    ip_info.append({"interface": "Hostname", "ip": hostname, "type": "System"})

    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip_info.append(
                    {
                        "interface": interface,
                        "ip": addr.address,
                        "type": "IPv4",
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast if addr.broadcast else "N/A",
                    }
                )
            elif addr.family == socket.AF_INET6:
                ip_info.append(
                    {
                        "interface": interface,
                        "ip": addr.address,
                        "type": "IPv6",
                        "netmask": addr.netmask,
                        "broadcast": "N/A",
                    }
                )

    return ip_info


def gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()

    bytes_to_mb = lambda x: x / (1024**2)
    memory_allocated = bytes_to_mb(torch.cuda.memory_allocated(device))
    memory_reserved = bytes_to_mb(torch.cuda.memory_reserved(device))
    peak_memory = bytes_to_mb(torch.cuda.max_memory_allocated())
    peak_reserved = bytes_to_mb(torch.cuda.max_memory_reserved())

    local_device_id = device
    global_device_id = local_device_id
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    param_info = {}
    for name, param in model.named_parameters():
        if param is not None and param.requires_grad:
            param_info[name] = {
                "device": str(param.device),
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            }
            break

    env_vars = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
    }

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        
        "parameter_sample": param_info,
        "env_vars": env_vars,
    }

