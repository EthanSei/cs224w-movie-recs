import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate torch device based on the specified device string.
    
    Args:
        device: Device specification. Options:
            - "auto": Automatically select CUDA if available, else CPU
            - "cuda": Use CUDA (raises error if not available)
            - "cpu": Use CPU
            - "cuda:N": Use specific CUDA device N
            - "mps": Use Apple Silicon GPU (if available)
    
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple MPS")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU available)")
    
    return torch.device(device)

