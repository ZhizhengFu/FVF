from src.config import Config
from .dataset import DefaultDataset
from torch.utils.data import DataLoader


def get_dataloader(opt: Config, mode: str = "train"):
    return DataLoader(
        DefaultDataset(opt.train) if mode == "train" else DefaultDataset(opt.test),
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.num_workers,
    )
