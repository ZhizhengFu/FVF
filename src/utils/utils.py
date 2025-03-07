import os
import sys
import torch
import wandb
import shutil
import random
import torchvision
import numpy as np
import torch.nn as nn
from typing import Dict
from pathlib import Path
from torch.nn import init
from functools import partial
from datetime import datetime
from itertools import zip_longest
from torch.optim.lr_scheduler import LRScheduler


COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"


class GradualWarmupScheduler(LRScheduler):
    pass


def save_code_snapshot(model_name: str, dir_name: str) -> None:
    src_dir = Path("core")
    dst_dir = Path("experiments") / model_name / dir_name
    for file_path in src_dir.rglob("*.py"):
        if file_path.is_file() and "__pycache__" not in file_path.parts:
            destination_path = dst_dir / file_path.relative_to(src_dir)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)
    shutil.copy(Path("configs") / f"{model_name}.toml", dst_dir)


def get_cur_time() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def init_wandb(project: str, name: str) -> None:
    wandb.init(project=project, name=name)


def init_seed(seed: int = 0, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _init_weights(
    module: nn.Module,
    *,
    init_type: str,
    bn_init_type: str,
    gain: float,
    a: float,
    nonlinearity: str,
    mean: float,
    std: float,
    bn_uniform_lower: float,
    bn_uniform_upper: float,
) -> None:
    """Internal weight initialization handler for different module types."""

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_type == "normal":
            init.normal_(module.weight, mean=mean, std=std)
        elif init_type == "xavier_normal":
            init.xavier_normal_(module.weight, gain=gain)
        elif init_type == "xavier_uniform":
            init.xavier_uniform_(module.weight, gain=gain)
        elif init_type == "kaiming_normal":
            init.kaiming_normal_(
                module.weight, a=a, mode="fan_in", nonlinearity=nonlinearity
            )
        elif init_type == "kaiming_uniform":
            init.kaiming_uniform_(
                module.weight, a=a, mode="fan_in", nonlinearity=nonlinearity
            )
        elif init_type == "orthogonal":
            init.orthogonal_(module.weight, gain=int(gain))
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        if module.bias is not None:
            init.constant_(module.bias, 0.0)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if bn_init_type == "default":
            init.constant_(module.weight, 1.0)
            init.constant_(module.bias, 0.0)
        elif bn_init_type == "uniform":
            init.uniform_(module.weight, a=bn_uniform_lower, b=bn_uniform_upper)
            init.constant_(module.bias, 0.0)
        elif bn_init_type == "normal":
            init.normal_(module.weight, mean=1.0, std=std)
            init.constant_(module.bias, 0.0)
        else:
            raise ValueError(f"Unsupported bn_init_type: {bn_init_type}")


def init_weights(
    net: nn.Module,
    init_type: str = "kaiming_normal",
    bn_init_type: str = "default",
    *,
    gain: float = 1.0,
    a: float = 0,
    nonlinearity: str = "leaky_relu",
    mean: float = 0.0,
    std: float = 0.02,
    bn_uniform_lower: float = 0.1,
    bn_uniform_upper: float = 1.0,
) -> None:
    """
    Initialize neural network weights with modern defaults and flexible options.

    Args:
        net: Neural network module to initialize
        init_type: Weight init scheme for Conv/Linear layers
            (normal/xavier_normal/xavier_uniform/kaiming_normal/kaiming_uniform/orthogonal)
        bn_init_type: Init scheme for normalization layers
            (default/uniform/normal)
        gain: Scaling factor for xavier/orthogonal inits
        a: Negative slope for kaiming inits
        nonlinearity: Nonlinearity for kaiming inits
        mean: Mean for normal distribution inits
        std: Std for normal distribution inits
        bn_uniform_lower: Lower bound for uniform batchnorm init
        bn_uniform_upper: Upper bound for uniform batchnorm init

    Example:
        >>> model = ResNet()
        >>> init_weights(model, init_type="kaiming_uniform", nonlinearity="relu")
    """
    if not isinstance(net, nn.Module):
        raise TypeError("net must be an instance of nn.Module")

    supported_init = {
        "normal",
        "xavier_normal",
        "xavier_uniform",
        "kaiming_normal",
        "kaiming_uniform",
        "orthogonal",
    }
    if init_type not in supported_init:
        raise ValueError(
            f"Invalid init_type: {init_type}. Choose from {supported_init}"
        )

    supported_bn_init = {"default", "uniform", "normal"}
    if bn_init_type not in supported_bn_init:
        raise ValueError(
            f"Invalid bn_init_type: {bn_init_type}. Choose from {supported_bn_init}"
        )

    init_fn = partial(
        _init_weights,
        init_type=init_type,
        bn_init_type=bn_init_type,
        gain=gain,
        a=a,
        nonlinearity=nonlinearity,
        mean=mean,
        std=std,
        bn_uniform_lower=bn_uniform_lower,
        bn_uniform_upper=bn_uniform_upper,
    )

    net.apply(init_fn)


def _get_time_components(delta) -> Dict[str, int]:
    total_seconds = delta.total_seconds()

    days = delta.days
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)

    hours, remainder = divmod(total_seconds % 86400, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "years": int(years),
        "months": int(months),
        "days": int(days),
        "hours": int(hours),
        "minutes": int(minutes),
        "seconds": int(seconds),
    }


def _format_duration(components: Dict[str, int]) -> str:
    return " ".join(f"{value}{unit}" for unit, value in components.items() if value > 0)


def print_env_info() -> None:
    time_since_first_commit = datetime.now() - datetime(2025, 1, 24, 0, 42, 0)
    time_components = _get_time_components(time_since_first_commit)

    versions = {
        "PyTorch": torch.__version__,
        "TorchVision": torchvision.__version__,
        "Python": sys.version.split()[0],
    }

    logo = [
        "\n    ████████╗  ██╗   ██╗  ████████╗",
        "    ██╔═════╝  ██║   ██║  ██╔═════╝",
        "    █████╗     ██║   ██║  █████╗",
        "    ██╔══╝     ╚██╗ ██╔╝  ██╔══╝",
        "    ██║         ╚████╔╝   ██║",
        "    ╚═╝          ╚═══╝    ╚═╝",
    ]

    version_info = "\n".join(
        ["\tVersion Information:"]
        + [
            f"\t{name}: {COLOR_GREEN}{ver}{COLOR_RESET}"
            for name, ver in versions.items()
        ]
    )

    time_info = (
        f"It's been this long since the {COLOR_GREEN}FVF{COLOR_RESET}'s first commit:\n"
        f"{_format_duration(time_components)}"
    )

    max_logo_width = max(map(len, logo)) + 4
    combined = [
        f"{COLOR_YELLOW}{logo_line.ljust(max_logo_width)}{COLOR_RESET}{text_line}"
        for logo_line, text_line in zip_longest(
            logo, f"{version_info}\n\n{time_info}".split("\n"), fillvalue=""
        )
    ]

    print("\n".join(combined))


if __name__ == "__main__":
    print(get_cur_time())
