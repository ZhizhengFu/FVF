import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pathlib import Path
from src.config import Config
from src.models import defaultnet
from src.data import get_dataloader
from src.utils import (
    Logger,
    get_cur_time,
    init_seed,
    save_code_snapshot,
    SSIM,
    PSNR,
)


class Trainer:
    def __init__(self, CONFIG_NAME: str, device: torch.device):
        self.CONFIG_NAME = CONFIG_NAME
        config = Config.from_toml(f"configs/{CONFIG_NAME}.toml")
        config.cur_time = get_cur_time()
        self.dst_dir = Path("experiments") / config.model.name / config.cur_time
        self.device = device
        self.opt = config.train
        self.best_val_loss = float("inf")
        self.psnr = PSNR().to(self.device)
        self.ssim = SSIM().to(self.device)
        self.logger = Logger(log_dir=self.dst_dir, config=config)
        self.train_dataloader = get_dataloader(config.datasets, "train", self.device)
        self.val_dataloader = get_dataloader(config.datasets, "val", self.device)
        self.test_dataloader = get_dataloader(config.datasets, "test", self.device)
        self._init_train()
        self.model = self._init_model(config.model, device)
        self.loss_fn = self._init_loss()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def run_loop(self):
        self.logger.log_message(f"Starting training with config: {self.CONFIG_NAME}")
        self.logger.log_message(f"Model architecture: {str(self.model)}")
        for epoch in range(
            self.opt.optimizer.warmup_epochs + self.opt.optimizer.decay_epochs
        ):
            train_loss, train_psnr, train_ssim = self._train_epoch(epoch)

            val_loss, val_psnr, val_ssim = self._validate()

            test_loss, test_psnr, test_ssim = self._test_model()
            metrics = {
                "train_loss": train_loss,
                "train_psnr": train_psnr,
                "train_ssim": train_ssim,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "test_loss": test_loss,
                "test_psnr": test_psnr,
                "test_ssim": test_ssim,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.logger.log_metrics(metrics, step=epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
                self.logger.log_message(
                    f"New best model at epoch {epoch} with val loss {val_loss:.4f}"
                )

            if epoch % self.opt.checkpoint_save == 0:
                self._save_checkpoint(epoch)

            self.scheduler.step()
            self.logger.finish()

    def _train_epoch(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        totlal_psnr = 0
        total_ssim = 0
        self.logger.log_message(f"Starting epoch {epoch}")

        for batch_idx, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.loss_fn(output, batch.H_img)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            totlal_psnr += self.psnr(output, batch.H_img).item()
            total_ssim += self.ssim(output, batch.H_img).item()

            if batch_idx % 1000 == 0:
                self.logger.log_message(
                    f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f} PSNR: {totlal_psnr / (batch_idx + 1):.4f} SSIM: {total_ssim / (batch_idx + 1):.4f}"
                )

        return (
            train_loss / len(self.train_dataloader),
            totlal_psnr / len(self.train_dataloader),
            total_ssim / len(self.train_dataloader),
        )

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        total_psnr = 0
        total_ssim = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                output = self.model(batch)[
                    ..., : batch.H_img.shape[2], : batch.H_img.shape[3]
                ]
                val_loss += self.loss_fn(output, batch.H_img).item()
                total_psnr += self.psnr(output, batch.H_img).item()
                total_ssim += self.ssim(output, batch.H_img).item()

        return (
            val_loss / len(self.val_dataloader),
            total_psnr / len(self.val_dataloader),
            total_ssim / len(self.val_dataloader),
        )

    def _test_model(self):
        self.model.eval()
        test_loss = 0.0
        total_psnr = 0
        total_ssim = 0

        with torch.no_grad():
            for batch in self.test_dataloader:
                output = self.model(batch)[
                    ..., : batch.H_img.shape[2], : batch.H_img.shape[3]
                ]
                test_loss += self.loss_fn(output, batch.H_img).item()
                total_psnr += self.psnr(output, batch.H_img).item()
                total_ssim += self.ssim(output, batch.H_img).item()

        return (
            test_loss / len(self.test_dataloader),
            total_psnr / len(self.test_dataloader),
            total_ssim / len(self.test_dataloader),
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        (self.dst_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
        }
        if is_best:
            torch.save(checkpoint, Path(self.dst_dir / "checkpoint" / "best_model.pth"))
        torch.save(
            checkpoint, Path(self.dst_dir / "checkpoint" / f"checkpoint_{epoch}.pth")
        )

    def _init_train(self) -> None:
        init_seed(self.opt.seed, self.opt.deterministic)
        if self.opt.save_snapshots:
            save_code_snapshot(self.dst_dir, self.CONFIG_NAME)

    @staticmethod
    def _init_model(opt: Config, device):
        model = defaultnet(opt).to(device)
        init_weights(
            model,
            init_type=opt.init_type,
            init_bn_type=opt.init_bn_type,
            gain=opt.init_gain,
            verbose=True,
        )
        return model

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


def init_weights(
    model: nn.Module,
    init_type: str = "normal",
    init_bn_type: str = "uniform",
    gain: float = 0.02,
    verbose: bool = False,
) -> None:
    """Initialize model weights with specified method."""

    # Initialize weight initializers first
    weight_initializer = _create_weight_initializer(init_type, gain)
    bn_initializer = _create_bn_initializer(init_bn_type, gain)

    def init_func(m: nn.Module) -> None:
        """Initialize a single layer based on its type."""
        if not hasattr(m, "weight"):
            return

        classname = m.__class__.__name__

        if _is_conv_or_linear(classname):
            weight_initializer(m)
            _init_bias(m)
            if verbose:
                print(f"Initialized {classname} (weight: {init_type}, bias: zero)")
        elif _is_batch_norm(classname):
            bn_initializer(m)
            if verbose:
                print(f"Initialized {classname} with {init_bn_type}")
        elif verbose:
            print(f"Layer {classname} has weights but no specific initialization")

    model.apply(init_func)


def _is_conv_or_linear(classname: str) -> bool:
    """Check if layer is convolutional or linear."""
    return classname.find("Conv") != -1 or classname.find("Linear") != -1


def _is_batch_norm(classname: str) -> bool:
    """Check if layer is batch normalization."""
    return classname.find("BatchNorm") != -1


def _init_bias(m: nn.Module) -> None:
    """Initialize bias terms to zero if they exist."""
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def _create_weight_initializer(init_type: str, gain: float):
    """Create a weight initializer function for conv/linear layers."""
    init_methods = {
        "normal": lambda m: init.normal_(m.weight.data, 0.0, gain),
        "xavier": lambda m: init.xavier_normal_(m.weight.data, gain=gain),
        "xavier_uniform": lambda m: init.xavier_uniform_(m.weight.data, gain=gain),
        "kaiming": lambda m: init.kaiming_normal_(
            m.weight.data, mode="fan_in", nonlinearity="relu"
        ),
        "orthogonal": lambda m: init.orthogonal_(
            m.weight.data, gain=int(gain) if gain is not None else 1
        ),
        "ones": lambda m: init.constant_(m.weight.data, 1.0),
        "zeros": lambda m: init.constant_(m.weight.data, 0.0),
    }

    if init_type not in init_methods:
        raise NotImplementedError(
            f"Initialization method [{init_type}] is not implemented"
        )

    return init_methods[init_type]


def _create_bn_initializer(init_bn_type: str, gain: float):
    """Create a batch norm initializer function."""

    def uniform_init(m: nn.Module) -> None:
        if m.weight is not None:
            init.uniform_(m.weight.data, 1.0 - gain, 1.0 + gain)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    def constant_init(m: nn.Module) -> None:
        if m.weight is not None:
            init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    init_methods = {
        "uniform": uniform_init,
        "constant": constant_init,
    }

    return init_methods.get(init_bn_type, lambda _: None)
