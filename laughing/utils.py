import torch
from pynvml import *

def check_gpu_memory():
    """Print out user-friendly GPU memory stats."""
    # Check allocated GPU memory in GB
    allocated_memory_bytes = torch.cuda.memory_allocated()
    allocated_memory_gb = allocated_memory_bytes / 1024**3  # Convert to GB
    print(f"Allocated GPU Memory: {allocated_memory_gb:.2f} GB")

    # Check maximum allocated GPU memory in GB
    max_memory_bytes = torch.cuda.max_memory_allocated()
    max_memory_gb = max_memory_bytes / 1024**3  # Convert to GB
    print(f"Maximum Allocated GPU Memory: {max_memory_gb:.2f} GB")

    available_memory_bytes = torch.cuda.get_device_properties(0).total_memory - allocated_memory_bytes
    available_memory_gb = available_memory_bytes / (1024**3)  # Convert to GB
    print(f"Available GPU Memory: {available_memory_gb:.2f} GB")

def free_gpu_memory():
    """Tries to free up GPU memory."""
    import gc
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    check_gpu_memory()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

