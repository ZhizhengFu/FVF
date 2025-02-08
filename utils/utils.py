import os
import random
import numpy as np
import torch
import warnings


def init_seed(seed=0, deterministic=False):
    """
    Initialize random seeds for reproducibility.

    :param seed: int, random seed value.
    :param deterministic: bool, if True, ensures deterministic behavior at the cost of performance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs

    if deterministic:
        _enable_deterministic_mode()
    else:
        _enable_performance_mode()


def _enable_deterministic_mode():
    """Enable deterministic mode for PyTorch."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # Disable cuDNN to enforce determinism
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        warnings.warn(
            f"Some operations may not have deterministic implementations in PyTorch: {e}",
            UserWarning,
        )
    warnings.warn(
        "Deterministic mode enabled, this may slow down training.", UserWarning
    )


def _enable_performance_mode():
    """Enable performance mode for PyTorch."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True  # Allow cuDNN optimizations


if __name__ == "__main__":
    init_seed(0, deterministic=False)
