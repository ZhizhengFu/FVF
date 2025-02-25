import os
import sys
import torch
import wandb
import shutil
import random
import datetime
import torchvision
import numpy as np
from pathlib import Path
from textwrap import dedent
from torch.optim.lr_scheduler import LRScheduler


class GradualWarmupScheduler(LRScheduler):
    pass

def save_code_snapshot(model_name: str, dir_name: str) -> None:
    src_dir = Path("core")
    dst_dir = Path("experiments") / model_name / dir_name
    for file_path in src_dir.rglob("*"):
        if file_path.is_file() and "__pycache__" not in file_path.parts:
            destination_path = dst_dir / file_path.relative_to(src_dir)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination_path)
    shutil.copy(Path("configs")/f"{model_name}.toml",dst_dir)

def get_cur_time() -> str:
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


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

def print_env_info() -> None:
    msg = r"""
    ████████╗  ██╗   ██╗  ████████╗
    ██╔═════╝  ██║   ██║  ██╔═════╝
    █████╗     ██║   ██║  █████╗
    ██╔══╝     ╚██╗ ██╔╝  ██╔══╝
    ██║         ╚████╔╝   ██║
    ╚═╝          ╚═══╝    ╚═╝
    """
    msg += dedent(f"""
        Version Information:
            PyTorch: {torch.__version__}
            TorchVision: {torchvision.__version__}
            Python: {sys.version}
    """)
    print(msg)

if __name__ == "__main__":
    print(get_cur_time())
