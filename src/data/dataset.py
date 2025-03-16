import random
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from src.config import Config
from src.utils import (
    imread_uint_3,
    inpaint_pipeline,
    mosaic_CFA_Bayer_pipeline,
    sisr_pipeline,
)


class DefaultDataset(Dataset):
    def __init__(self, opt: Config):
        self.transform = self._get_transform(opt)
        self.img_paths = list(self._get_root_dir(opt).glob("*"))
        self.processing_func = [
            inpaint_pipeline,
            mosaic_CFA_Bayer_pipeline,
            sisr_pipeline,
        ]

    @staticmethod
    def _get_root_dir(opt: Config):
        if opt.root_dir:
            return Path(opt.root_dir)
        else:
            raise ValueError("root_dir is not specified")

    @staticmethod
    def _get_transform(opt: Config):
        if opt.mode == "train":
            patch_size = opt.patch_size if opt.patch_size else 96
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
        else:
            return None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = imread_uint_3(self.img_paths[idx])
        if self.transform:
            image = self.transform(image=image)["image"]
        return random.choice(self.processing_func)(image)
