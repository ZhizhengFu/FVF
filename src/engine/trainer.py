import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pathlib import Path
from src.config import Config
from src.models import defaultnet
from src.data import get_dataloader
from src.utils import (
    get_cur_time,
    init_wandb,
    init_seed,
    save_code_snapshot,
)


class Trainer:
    def __init__(self, CONFIG_NAME: str, device: torch.device):
        self.CONFIG_NAME = CONFIG_NAME
        config = Config.from_toml(f"configs/{CONFIG_NAME}.toml")
        self.opt = config.train
        self.device = device
        self.dst_dir = (
            Path("experiments") / config.model.name / self._generate_dir_name()
        )
        self._init_train()
        self.train_dataloader = get_dataloader(config.datasets, "train", self.device)
        self.model = self._init_model(config.model)
        self.loss_fn = self._init_loss()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def run_loop(self):
        for epoch in range(
            self.opt.optimizer.warmup_epochs + self.opt.optimizer.decay_epochs
        ):
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(batch), batch.H_img)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            if epoch % self.opt.checkpoint_save == 0:
                self._save_checkpoint(epoch)
            # if epoch % self.opt.log_interval == 0:
            #     self._log_metrics(epoch)
            # if epoch % self.opt.checkpoint_test == 0:
            #     self._test_model()

    def _save_checkpoint(self, epoch):
        (self.dst_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
            },
            Path(self.dst_dir / "checkpoint" / f"checkpoint_{epoch}.pth"),
        )

    def _init_train(self) -> None:
        init_seed(self.opt.seed, self.opt.deterministic)
        if self.opt.save_snapshots:
            save_code_snapshot(self.dst_dir, self.CONFIG_NAME)
        if self.opt.wandb.is_enabled:
            init_wandb(self.opt.wandb.project, self._generate_wandb_name())

    @staticmethod
    def _generate_wandb_name() -> str:
        return get_cur_time()

    @staticmethod
    def _generate_dir_name(info: str | None = None) -> str:
        return get_cur_time() + (str(info) if info else "")

    @staticmethod
    def _init_model(opt: Config):
        return defaultnet(opt)

    def _init_loss(self):
        if self.opt.ls_fn == "l1":
            return nn.L1Loss().to(self.device)
        elif self.opt.ls_fn == "mse":
            return nn.MSELoss().to(self.device)
        else:
            raise NotImplementedError

    def _init_optimizer(self):
        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print("Params [{:s}] will not optimize.".format(k))
        if self.opt.optimizer.name == "adam":
            return torch.optim.Adam(optim_params, lr=self.opt.optimizer.lr)
        else:
            raise NotImplementedError

    def _init_scheduler(self):
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.opt.optimizer.start_factor,
            total_iters=self.opt.optimizer.warmup_epochs,
        )
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.opt.optimizer.decay_epochs,
            eta_min=self.opt.optimizer.start_factor * self.opt.optimizer.lr,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.opt.optimizer.warmup_epochs],
        )
