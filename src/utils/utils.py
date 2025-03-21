import os
import sys
import torch
import wandb
import shutil
import random
import torchvision
import numpy as np
from enum import Enum
from typing import Dict
from pathlib import Path
from datetime import datetime
from itertools import zip_longest


class Color(Enum):
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    def apply(self, text: str) -> str:
        return f"{self.value}{text}{Color.RESET.value}"


def save_code_snapshot(model_name: str, dir_name: str, config_name: str) -> None:
    src_dir = Path("src")
    dst_dir = Path("experiments") / model_name / dir_name
    for file_path in src_dir.rglob("*.py"):
        if file_path.is_file() and "__pycache__" not in file_path.parts:
            destination_path = dst_dir / file_path.relative_to(src_dir)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)
    shutil.copy(Path("configs") / f"{config_name}.toml", dst_dir)


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
    time_since_first_commit = datetime.now() - datetime(2025, 3, 15, 3, 12, 0)
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
        + [f"\t{name}: {Color.GREEN.apply(ver)}" for name, ver in versions.items()]
    )

    time_info = (
        f"It's been this long since the {Color.GREEN.apply('FVF')}'s first commit:\n"
        f"{_format_duration(time_components)}"
    )

    max_logo_width = max(map(len, logo)) + 4
    combined = [
        f"{Color.YELLOW.apply(logo_line.ljust(max_logo_width))}{text_line}"
        for logo_line, text_line in zip_longest(
            logo, f"{version_info}\n\n{time_info}".split("\n"), fillvalue=""
        )
    ]

    print("\n".join(combined))
