import torch
import torch.utils.data as data
import random
import numpy as np
import albumentations as A
from typing import List
from pathlib import Path
from functools import partial
from numpy.typing import NDArray
from src.config import Config
from src.utils import (
    imread_uint_3,
    sisr_pipeline,
    mosaic_CFA_Bayer_pipeline,
    inpaint_pipeline,
    DegradationOutput,
)


class DefaultDataset(data.Dataset):
    def __init__(self, opt: Config, mode: str, device: torch.device):
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.device = device
        self.transform = self._get_transform()
        self.img_paths = list(self._get_root_dir().glob("*"))

    def _get_root_dir(self) -> Path:
        if self.opt.root_dir:
            return Path(self.opt.root_dir)
        else:
            raise ValueError("root_dir is not specified")

    def _get_transform(self) -> A.Compose | None:
        if self.mode != "train":
            return None
        patch_size = self.opt.patch_size if self.opt.patch_size else 96
        return A.Compose(
            [
                A.Rotate(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2
                ),
                A.RandomCrop(height=patch_size, width=patch_size),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx) -> NDArray[np.uint8]:
        image = imread_uint_3(self.img_paths[idx])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image

    @staticmethod
    def collate_fn(
        batch: List[NDArray[np.uint8]], opt: Config, device: torch.device
    ) -> DegradationOutput:
        sf = random.choice(opt.sf)
        sr = random.uniform(opt.sr[0], opt.sr[1])
        pipeline = random.choice(
            [
                lambda img: mosaic_CFA_Bayer_pipeline(img, device),
                lambda img: sisr_pipeline(img, sf, device),
                lambda img: inpaint_pipeline(img, sr, device),
            ]
        )
        outputs = [pipeline(img) for img in batch]

        H_img_batch = torch.stack([output.H_img for output in outputs])
        L_img_batch = torch.stack([output.L_img for output in outputs])
        R_img_batch = torch.stack([output.R_img for output in outputs])
        mask_batch = torch.stack([output.mask for output in outputs])
        k_batch = torch.stack([output.k for output in outputs])
        sigma_batch = torch.stack([output.sigma for output in outputs])
        sr_batch = outputs[0].sr
        sf_batch = outputs[0].sf

        return DegradationOutput(
            H_img=H_img_batch,
            L_img=L_img_batch,
            R_img=R_img_batch,
            mask=mask_batch,
            k=k_batch,
            sigma=sigma_batch,
            sr=sr_batch,
            sf=sf_batch,
        )

    def get_collate_fn(self):
        return partial(self.collate_fn, opt=self.opt, device=self.device)
