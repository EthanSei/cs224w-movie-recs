import gc
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


def clear_memory():
    """
    Clear GPU memory cache and run garbage collection.
    
    Call this between training and evaluation, or between tuning trials
    to free up unused GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't have empty_cache, but gc.collect helps
        pass


def log_memory_usage(prefix: str = ""):
    """
    Log current GPU memory usage for debugging OOM issues.
    
    Args:
        prefix: Optional prefix for the log message
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"{prefix}GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB")

