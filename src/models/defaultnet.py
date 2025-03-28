import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
from src.config import Config
from src.utils import DegradationOutput
from .backbone import ResUNet


class HyperNet(nn.Module):
    def __init__(self, in_nc=4, channel=64, out_nc=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


class DataNet(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def splits_and_mean(a, sf):
        b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
        b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
        return torch.mean(b, dim=-1)

    def forward(self, x, FK, FCK, F2K, FCKFSHy, alpha, sf, mask):
        FR = FCKFSHy + fft2(alpha * x)
        _FKFR_, _F2K_ = self.splits_and_mean(FK * FR, sf), self.splits_and_mean(F2K, sf)
        _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
        FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(1, 1, sf, sf)
        FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / alpha
        return torch.real(ifft2(FX)) * (alpha + 1) / (alpha + mask)


class defaultnet(nn.Module):
    def __init__(self, opt: Config):
        super().__init__()
        self.opt = opt
        self.d = DataNet()
        self.p = ResUNet()
        self.h = HyperNet(in_nc=2, channel=64, out_nc=2 * opt.iter_num)

    @staticmethod
    def prepare_frequency_components(input: DegradationOutput) -> tuple:
        K = torch.zeros_like(input.R_img)
        K[..., : input.k.size(-1), : input.k.size(-1)].copy_(input.k)
        K = torch.roll(
            K, (-(torch.tensor(input.k.shape[-2:]) // 2)).tolist(), dims=(2, 3)
        )
        FK = fft2(K)
        FCK, F2K = torch.conj(FK), torch.abs(FK) ** 2
        # [todo] SHy = input.R_img*input.mask
        SHy = torch.zeros_like(input.R_img)
        SHy[..., 0::input.sf, 0::input.sf] = input.L_img
        FCKFSHy = FCK * fft2(SHy)
        return FK, FCK, F2K, FCKFSHy

    def forward(self, input: DegradationOutput):
        FK, FCK, F2K, FCKFSHy = self.prepare_frequency_components(input)
        for i in range(self.opt.iter_num):
            ab = self.h(
                torch.cat((input.sigma,torch.tensor(input.sr).type_as(input.sigma).expand_as(input.sigma),
                        # torch.tensor(i + 1).type_as(input.sigma).expand_as(input.sigma),
                    ),dim=1,)
            )
            input.R_img = self.d(
                input.R_img,
                FK,
                FCK,
                F2K,
                FCKFSHy,
                ab[:, i : i + 1, ...],
                input.sf,
                input.mask,
            )
            input.R_img = self.p(
                torch.cat(
                    (
                        input.R_img,
                        ab[:, self.opt.iter_num + i : self.opt.iter_num + i + 1, ...]
                        .unsqueeze(1)
                        .repeat(1, 1, input.R_img.size(2), input.R_img.size(3)),
                    ),
                    dim=1,
                )
            )
        return input
