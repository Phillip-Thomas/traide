import os
import torch
import torch.distributed as dist
import psutil
import multiprocessing
import numpy as np

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_device():
    """Get the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Verify MPS is actually working
        try:
            torch.zeros(1).to('mps')
            return "mps"
        except:
            print("MPS (Metal) is available but not working, falling back to CPU")
            return "cpu"
    return "cpu"

def get_optimal_batch_size(device):
    """Determine optimal batch size based on available hardware"""
    if device.type == "cuda":
        # Get GPU memory info
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        mem_gb = gpu_mem / (1024**3)  # Convert to GB

        # More aggressive batch sizing
        base_batch = int(min(2048, max(256, (mem_gb / 4) * 256)))
        batch_size = 2 ** int(np.log2(base_batch))

        print(f"GPU Memory: {mem_gb:.1f}GB")
        print(f"Optimal batch size: {batch_size}")
        return batch_size

    elif device.type == "mps":
        # For Apple Silicon, use system memory as proxy
        mem_gb = psutil.virtual_memory().total / (1024**3)
        batch_size = min(256, max(32, int((mem_gb / 16) * 128)))
        print(f"System Memory: {mem_gb:.1f}GB")
        print(f"Optimal batch size: {batch_size}")
        return batch_size

    else:  # CPU
        # Use number of CPU cores to determine batch size
        num_cores = multiprocessing.cpu_count()
        batch_size = min(128, max(16, num_cores * 8))
        print(f"CPU Cores: {num_cores}")
        print(f"Optimal batch size: {batch_size}")
        return batch_size 