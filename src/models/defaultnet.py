import torch.nn as nn
from src.config import Config
from src.utils import DegradationOutput


class defaultnet(nn.Module):
    def __init__(self, opt: Config):
        super().__init__()
        self.opt = opt
        self.net = nn.Conv2d(3, 32, kernel_size=3, padding=1)

    def forward(self, input: DegradationOutput):
        pass
