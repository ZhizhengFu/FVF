import torch
import torch.utils.data as data
from typing import Literal
from src.config import Config
from .dataset import DefaultDataset


def get_dataloader(
    opt: Config, mode: Literal["train", "test"], device: torch.device
) -> data.DataLoader:
    dataset = (
        DefaultDataset(opt.train, mode, device)
        if mode == "train"
        else DefaultDataset(opt.test, mode, device)
    )
    return data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        pin_memory=opt.pin_memory,
        num_workers=opt.num_workers,
        collate_fn=dataset.get_collate_fn(),
        shuffle=True if mode == "train" else False,
    )
