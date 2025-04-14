import torch
import numpy as np

def ensure_tensor_on_device(tensor_or_array, device):
    """Ensure a tensor or array is on the specified device."""
    if isinstance(tensor_or_array, np.ndarray):
        return torch.tensor(tensor_or_array, device=device)
    elif isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.to(device)
    else:
        return torch.tensor(tensor_or_array, device=device)
