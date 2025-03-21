import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
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
        self.opt = Config.from_toml(f"configs/{CONFIG_NAME}.toml")
        self.device = device
        self._init_train()
        self.train_dataloader = get_dataloader(self.opt.datasets, "train", self.device)
        self.model = self._init_model(self.opt.model)
        self.loss_fn = self._init_loss()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def run_loop(self):
        for epoch in range(
            self.opt.train.optimizer.warmup_epochs
            + self.opt.train.optimizer.decay_epochs
        ):
            print(f"Epoch {epoch}")
            for batch in self.train_dataloader:
                # self.optimizer.zero_grad()
                # loss = self.loss_fn(self.model(batch), batch.H_img)
                # loss.backward()
                self.optimizer.step()
            self.scheduler.step()

    def _init_train(self) -> None:
        init_seed(self.opt.train.seed, self.opt.train.deterministic)
        if self.opt.train.save_snapshots:
            save_code_snapshot(
                self.opt.model.name, self._generate_dir_name(), self.CONFIG_NAME
            )
        if self.opt.train.wandb.is_enabled:
            init_wandb(self.opt.train.wandb.project, self._generate_wandb_name())

    @staticmethod
    def _generate_wandb_name() -> str:
        return get_cur_time()

    def _generate_dir_name(self) -> str:
        return f"{get_cur_time()}_{self.opt.train.optimizer.lr:.0e}"

    @staticmethod
    def _init_model(opt: Config):
        return defaultnet(opt)

    def _init_loss(self):
        if self.opt.train.ls_fn == "l1":
            return torch.nn.L1Loss().to(self.device)
        elif self.opt.train.ls_fn == "mse":
            return torch.nn.MSELoss().to(self.device)
        else:
            raise NotImplementedError

    def _init_optimizer(self):
        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print("Params [{:s}] will not optimize.".format(k))
        if self.opt.train.optimizer.name == "adam":
            return torch.optim.Adam(optim_params, lr=self.opt.train.optimizer.lr)
        else:
            raise NotImplementedError

    def _init_scheduler(self):
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.opt.train.optimizer.start_factor,
            total_iters=self.opt.train.optimizer.warmup_epochs,
        )
        decay_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.opt.train.optimizer.decay_epochs,
            eta_min=self.opt.train.optimizer.start_factor * self.opt.train.optimizer.lr,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.opt.train.optimizer.warmup_epochs],
        )
