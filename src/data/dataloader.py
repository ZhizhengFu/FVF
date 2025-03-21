import torch
from typing import Literal
from src.config import Config
from .dataset import DefaultDataset
from torch.utils.data import DataLoader


def get_dataloader(
    opt: Config, mode: Literal["train", "test"], device: torch.device
) -> DataLoader:
    dataset = (
        DefaultDataset(opt.train, mode, device)
        if mode == "train"
        else DefaultDataset(opt.test, mode, device)
    )
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        collate_fn=dataset.get_collate_fn(),
        shuffle=True if mode == "train" else False,
    )
