import torch
from src.config import Config
from src.utils import (
    get_cur_time,
    init_wandb,
    init_seed,
    save_code_snapshot,
)
from src.data import get_dataloader


class Trainer:
    def __init__(self, opt: Config, device: torch.device):
        self.opt: Config = opt
        self.device = device
        self._init_train()

    def run_loop(self):
        print("Running loop")

    def _init_train(self):
        init_seed(self.opt.train.seed, self.opt.train.deterministic)
        if self.opt.train.save_snapshots:
            save_code_snapshot(self.opt.model.name, self._generate_dir_name())
        if self.opt.train.wandb.is_enabled:
            init_wandb(self.opt.train.wandb.project, self._generate_wandb_name())
        self.train_dataloader = get_dataloader(self.opt.datasets, mode="train")

    def _generate_wandb_name(self) -> str:
        return get_cur_time()

    def _generate_dir_name(self) -> str:
        return f"{get_cur_time()}_{self.opt.train.lr:.0e}"
