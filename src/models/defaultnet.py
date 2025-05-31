import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
import torch.nn.functional as F
from src.config import Config
from src.utils import DegradationOutput, wiener_denoiser
from .backbone import ResUNet
from src.utils.utils_image import imshow, tensor2float

class HyperNet(nn.Module):
    def __init__(self, in_nc=2, channel=64, out_nc=8):
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
        return self.net(x) + 1e-7


class DataNet(nn.Module):
    def __init__(self, need_loop=True):
        super().__init__()
        self.need_loop = need_loop
        self.mode = "bicubic"

    @staticmethod
    def splits_and_mean(a, sf):
        b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
        b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
        return torch.mean(b, dim=-1)

    def forward(self, input: DegradationOutput, alpha, FK, FCK, F2K, FCKFSHy):
        if input.type == 1:
            sf_ = input.sf
            v = input.R_img.clone()
            while True:
                FR = FCKFSHy + fft2(alpha * v)
                _FKFR_, _F2K_ = (
                    self.splits_and_mean(FK * FR, input.sf),
                    self.splits_and_mean(F2K, input.sf),
                )
                _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
                FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(
                    1, 1, input.sf, input.sf
                )
                FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / alpha
                ans_ = ifft2(FX).real

                imshow([tensor2float(ans_.squeeze())])
                if (sf_ == 1) or (sf_ == 2) or (sf_ == 3) or (not self.need_loop):
                    break
                sf_ = sf_ // 2
                SHx_n_true = F.avg_pool2d(ans_, kernel_size=sf_, stride=sf_)
                v = F.interpolate(
                    wiener_denoiser(SHx_n_true, input.sigma),
                    scale_factor=sf_,
                    mode=self.mode,
                )

                print(sf_)
                imshow([tensor2float(ans_.squeeze()), tensor2float(SHx_n_true.squeeze())])
                # breakpoint()
            return ans_
        else:
            return (input.L_img + alpha * input.R_img) / (input.mask + alpha)


class defaultnet(nn.Module):
    def __init__(self, opt: Config):
        super().__init__()
        self.opt = opt
        self.d = DataNet()
        self.p = ResUNet()
        self.h = HyperNet(in_nc=2, channel=64, out_nc=2 * opt.iter_num)

    @staticmethod
    def prepare_frequency_components(input: DegradationOutput) -> tuple:
        K = torch.flip(input.k, [-1, -2])
        FK = fft2(K, s=(input.R_img.size(-2), input.R_img.size(-1)))
        FCK, F2K = torch.conj(FK), torch.abs(FK) ** 2
        SHy = torch.zeros_like(input.R_img)
        SHy[..., 0 :: input.sf, 0 :: input.sf] = wiener_denoiser(
            input.L_img, input.sigma
        )
        FCKFSHy = FCK * fft2(SHy)
        return FK, FCK, F2K, FCKFSHy

    def forward(self, input: DegradationOutput):
        if input.type == 1:
            FK, FCK, F2K, FCKFSHy = self.prepare_frequency_components(input)
        else:
            FK, FCK, F2K, FCKFSHy = None, None, None, None
        ab = self.h(
            torch.cat(
                (
                    input.sigma,
                    torch.tensor(input.sf).type_as(input.sigma).expand_as(input.sigma),
                ),
                dim=1,
            )
        )
        ans = None
        imshow([tensor2float(input.L_img.squeeze()),tensor2float(input.H_img.squeeze()), tensor2float(input.R_img.squeeze()), (input.k.squeeze())])
        for i in range(self.opt.iter_num):
            ab[:, i : i + 1, ...]=4e-6
            ans = self.d(
                input,
                ab[:, i : i + 1, ...],
                FK,
                FCK,
                F2K,
                FCKFSHy,
            )
            # ans = self.p(
            #     torch.cat(
            #         (
            #             ans,
            #             ab[
            #                 :, self.opt.iter_num + i : self.opt.iter_num + i + 1, ...
            #             ].repeat(1, 1, input.R_img.size(-2), input.R_img.size(-1)),
            #         ),
            #         dim=1,
            #     )
            # )
        return ans
