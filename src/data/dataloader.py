import torch
import torch.utils.data as data
from typing import Literal
from src.config import Config
from .dataset import DefaultDataset


def get_dataloader(
    opt: Config, mode: Literal["train", "val", "test"], device: torch.device
) -> data.DataLoader:
    if mode == "train":
        dataset = DefaultDataset(opt.train, mode, device)
    elif mode == "val":
        dataset = DefaultDataset(opt.val, mode, device)
    elif mode == "test":
        dataset = DefaultDataset(opt.test, mode, device)
    return data.DataLoader(
        dataset,
        batch_size=opt.batch_size if mode == "train" else 1,
        pin_memory=opt.pin_memory,
        num_workers=opt.num_workers,
        collate_fn=dataset.get_collate_fn(),
        shuffle=True if mode == "train" else False,
    )
