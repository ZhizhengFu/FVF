from src.config import Config
from src.utils import (
    get_cur_time,
    init_wandb,
    init_seed,
    save_code_snapshot,
)


class Trainer:
    def __init__(self, config: Config):
        self.config: Config = config
        self._init_train()

    def run_loop(self):
        print("Running loop")

    def _init_train(self):
        init_seed(self.config.train.seed, self.config.train.deterministic)
        if self.config.train.save_snapshots:
            save_code_snapshot(self.config.model.name, self._generate_dir_name())
        if self.config.train.wandb.is_enabled:
            init_wandb(self.config.train.wandb.project, self._generate_wandb_name())

    def _generate_wandb_name(self) -> str:
        return get_cur_time()

    def _generate_dir_name(self) -> str:
        return f"{get_cur_time()}_{self.config.train.lr:.0e}"
