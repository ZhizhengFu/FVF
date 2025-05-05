import torch
import torch.fft
from src.utils.utils_image import wiener_denoiser, PSNR
import torch
import torch.nn.functional as F
# from kornia.filters import median_blur


def avg_pool2d_reflect_padding(input_tensor, sf):
    # padded = F.pad(input_tensor, pad=(0, 1, 0, 1), mode='reflect')
    output = F.avg_pool2d(input_tensor, kernel_size=sf, stride=sf, padding=0)
    return output


def show_freq(x):
    m = torch.log(1e-6 + torch.abs(torch.fft.fft2(x).squeeze()))
    return (m - m.min()) / (m.max() - m.min())


def splits_and_mean(a, sf):
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return torch.mean(b, dim=-1)


def circular_conv_2d_fft(image: torch.Tensor, kernel: torch.Tensor):
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, Kh, Kw)
    batch_size, channels, h, w = image.shape
    kh, kw = kernel.shape[-2:]

    kernel = torch.flip(kernel, [-1, -2])
    fft_image = torch.fft.fft2(image)
    fft_kernel = torch.fft.fft2(kernel, s=(h, w))
    product = fft_image * fft_kernel
    result = torch.fft.ifft2(product)
    pad_h = kh - 1
    pad_w = kw - 1
    result_true = torch.roll(result, [-(pad_h // 2), -(pad_w // 2)], dims=(-2, -1))

    return (
        torch.real(result),
        (fft_image),
        (fft_kernel),
        (product),
        torch.real(result_true),
    )


if __name__ == "__main__":
    from src.utils.utils_image import (
        imshow,
        imread_uint_3,
        uint2tensor,
        tensor2float,
        KernelSynthesizer,
    )
    from pathlib import Path
    import torch.nn.functional as F
    import cv2
    import numpy as np

    imsize = 300
    image = torch.zeros(1, 3, imsize, imsize)
    center_h = (imsize) // 2
    center_w = (imsize) // 2
    www = 1
    image[:, :, center_h - www : center_h + www + 1, :] = 1
    image[:, :, :, center_w - www : center_w + www + 1] = 1
    radius = 70
    ring_width = 9
    h, w = image.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    distance = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
    circle_mask = (distance >= radius - ring_width / 2) & (
        distance <= radius + ring_width / 2
    )
    image[:, :, circle_mask] = 1
    # image = uint2tensor(imread_uint_3(Path("/Users/fuzz/Desktop/ILSVRC2012_val_00000001.jpg"))).unsqueeze(0)
    image = uint2tensor(imread_uint_3(Path("src/utils/test.png"))).unsqueeze(0)

    torch.manual_seed(40)
    psnr = PSNR()
    need_denoiser = True
    need_loop = True
    kernel = KernelSynthesizer().gen_motion_kernel()
    pad_h, pad_w = kernel.size(-2) - 1, kernel.size(-1) - 1
    sf = 4
    image = image[..., : image.shape[-2] // sf * sf, : image.shape[-1] // sf * sf]
    sigma = 10
    mode = "nearest"
    alpha = 6e-4

    Hx, Fx, Fk, FHx, Hx_true = circular_conv_2d_fft(image, kernel)
    SHy = torch.zeros_like(Hx)
    SHx = Hx[..., ::sf, ::sf]
    noise = torch.randn_like(SHx) * sigma / 255.0
    SHx_n = SHx + noise
    SHx_n_true = Hx_true[..., ::sf, ::sf] + noise
    SHy[..., ::sf, ::sf] = wiener_denoiser(SHx_n, sigma) if need_denoiser else SHx_n

    # imshow([tensor2float(image.squeeze()), tensor2float(SHx_n.squeeze()), tensor2float(SHx_n_true.squeeze())])
    while True:
        print(sf)
        v = (
            torch.nn.functional.interpolate(
                wiener_denoiser(SHx_n_true, sigma), scale_factor=sf, mode=mode
            )
            if need_denoiser
            else torch.nn.functional.interpolate(SHx_n_true, scale_factor=sf, mode=mode)
        )
        print(torch.std(v - image), psnr(v, image))

        sf_ = sf
        sf = 4

        K = torch.flip(kernel, [-1, -2])
        FK = (
            torch.fft.fft2(K.to(torch.float64), s=(image.size(-2), image.size(-1)))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        v, alpha = v.to(torch.float64), alpha
        FCK, F2K = torch.conj(FK), torch.abs(FK) ** 2
        FCKFSHy = FCK * torch.fft.fft2(SHy.to(torch.float64))
        FR = FCKFSHy + torch.fft.fft2(alpha * v)
        _FKFR_, _F2K_ = splits_and_mean(FK * FR, sf), splits_and_mean(F2K, sf)
        _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
        FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(1, 1, sf, sf)
        FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / alpha

        ans_ = torch.real(torch.fft.ifft2(FX))
        # ans_ = median_blur(ans_, kernel_size=9).clamp(0, 1)
        sf = sf_

        if sf == 2 or sf == 3 or not need_loop:
            break
        sf = sf // 2
        SHx_n_true = avg_pool2d_reflect_padding(ans_, sf)
        print(torch.std(ans_ - image), psnr(ans_, image))
    # imshow([tensor2float(ans_.squeeze()), tensor2float(SHx_n.squeeze()), tensor2float(SHx_n_true.squeeze())])

    # mask = torch.flip(K,[-1, -2])
    # my_mask = torch.zeros_like(ans_)
    # for i in range(0, ans_.size(-2)//sf):
    #     for j in range(0, ans_.size(-1)//sf):
    #         for u in range(0, mask.size(-2)):
    #             for e in range(0, mask.size(-1)):
    #                 my_mask[...,(i*sf+u)%ans_.size(-2):(i*sf+u+1)%ans_.size(-2), (j*sf+e)%ans_.size(-1):(j*sf+e+1)%ans_.size(-1)] += mask[...,u:u+1,e:e+1]
    # my_mask *= 30
    # imshow([tensor2float(my_mask.squeeze())])
    print(torch.std(ans_ - image), psnr(ans_, image))
    breakpoint()
    imshow(
        [
            tensor2float(image.squeeze()),
            tensor2float(Hx.squeeze()),
            tensor2float(SHx.squeeze()),
            tensor2float(SHx_n.squeeze()),
            tensor2float(SHy.squeeze()),
            tensor2float(show_freq(image)),
            tensor2float(show_freq(Hx)),
            tensor2float(show_freq(SHx)),
            tensor2float(show_freq(SHx_n)),
            tensor2float(show_freq(SHy)),
            tensor2float(v.squeeze()),
            tensor2float(show_freq(v)),
            tensor2float(ans_.squeeze()),
            # tensor2float(show_freq(ans_)),
            # Fk.squeeze().unsqueeze(-1).repeat(1,1,3), # type: ignore[arg-type]
            # K>0.004, # type: ignore[arg-type]
            tensor2float(torch.abs(image.squeeze() - ans_.squeeze())),
            tensor2float(torch.abs(ans_.squeeze() - v.squeeze())),
        ],
        [
            "x",
            "Hx",
            "SHx",
            "SHx_n",
            "SHy",
            "Fx",
            "FHx",
            "SHx_fft",
            "SHx_n_fft",
            "SHy_fft",
            "v",
            "v_fft",
            "ans",
            "ans_fft",
            "K",
        ],
    )
