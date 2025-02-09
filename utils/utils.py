import os
import torch
import random
import warnings
import numpy as np


def init_seed(seed=0, n_gpu=1, deterministic=False):
    """
    Initialize random seeds for reproducibility.

    :param seed: int, random seed value.
    :param n_gpu: int, number of GPUs.
    :param deterministic: bool, if True, ensures deterministic behavior at the cost of performance.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    if not isinstance(n_gpu, int) or n_gpu < 1:
        raise ValueError("Number of GPUs must be a positive integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
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
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


if __name__ == "__main__":
    init_seed(0, deterministic=False)
