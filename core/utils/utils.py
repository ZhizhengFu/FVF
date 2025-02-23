import os
import torch
import wandb
import shutil
import random
import datetime
import subprocess
import numpy as np
from pathlib import Path
from torch.optim.lr_scheduler import LRScheduler


class GradualWarmupScheduler(LRScheduler):
    pass


def save_code_snapshot(model_name: str, dir_name: str) -> None:
    dir_path = Path("experiments") / model_name / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    for file_path in _get_git_files("git ls-files"):
        source_path = Path(file_path)
        if source_path.is_dir():
            continue
        destination_path = dir_path / file_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)


def _get_git_files(command: str) -> set[str]:
    output = subprocess.check_output(command, shell=True)
    return set(output.decode().splitlines())


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


if __name__ == "__main__":
    print(get_cur_time())
