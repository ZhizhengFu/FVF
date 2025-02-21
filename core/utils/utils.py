import os
import wandb
import datetime
import random
import torch
import numpy as np
from typing import Any


def get_value(my_dict: dict, key: str, default: Any = None) -> Any:
    value = my_dict.get(key, default)
    return value if value != "" else default


def get_cur_time() -> str:
    return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


def create_experiment_directory(model_name: str, dir_name):
    dir_path = os.path.join("experiments", model_name, dir_name)
    os.makedirs(dir_path, exist_ok=True)


def init_wandb(project: str, name: str):
    wandb.init(project=project, name=name)


def init_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    print(get_cur_time())
