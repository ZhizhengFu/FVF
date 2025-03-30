import wandb
import logging
from typing import Dict
from pathlib import Path
from src.config import Config
from src.utils import init_wandb


class Logger:
    def __init__(self, log_dir: Path, config: Config):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_file_logging()
        self.wandb_enabled = config.train.wandb.is_enabled
        if self.wandb_enabled:
            init_wandb(config.train.wandb.project, config.cur_time)

    def _init_file_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("trainer")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        metrics_str = ", ".join(f"{k}: {v:.5f}" for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")
        if self.wandb_enabled:
            wandb.log(metrics, step=step)

    def log_message(self, message: str):
        self.logger.info(message)

    def finish(self):
        if self.wandb_enabled:
            wandb.finish()
