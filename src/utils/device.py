import torch

def is_cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def is_mps_available():
    """Check if MPS (Metal Performance Shaders) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def get_device():
    """Get the best available device for training."""
    if is_cuda_available():
        return torch.device('cuda')
    elif is_mps_available():
        return torch.device('mps')
    return torch.device('cpu') 