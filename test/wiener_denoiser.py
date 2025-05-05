import torch
import torch.fft
from src.utils import PSNR


def wiener_denoiser(SHx_n, sigma):
    sigma_norm = sigma / 255.0
    noise_power = sigma_norm**2
    signal_fft = torch.fft.fft2(SHx_n)
    signal_power = torch.abs(signal_fft) ** 2 / torch.numel(SHx_n)
    wiener_filter = signal_power / (signal_power + noise_power)
    denoised_fft = signal_fft * wiener_filter
    denoised = torch.fft.ifft2(denoised_fft).real
    return denoised, wiener_filter


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

    return (
        torch.real(result),
        torch.abs(fft_image),
        torch.abs(fft_kernel),
        torch.abs(product),
    )


if __name__ == "__main__":
    torch.manual_seed(55)
    from src.utils.utils_image import (
        imshow,
        imread_uint_3,
        uint2tensor,
        tensor2float,
        KernelSynthesizer,
    )
    from pathlib import Path

    image = uint2tensor(imread_uint_3(Path("src/utils/test.png"))).unsqueeze(0)
    kernel = KernelSynthesizer().gen_gaussian_kernel()
    sf = 3
    image = image[..., : image.shape[-1] // sf * sf, : image.shape[-2] // sf * sf]
    sigma = 7.65
    mode = "nearest"
    alpha = 5e-4
    limi = 1e-2

    Hx, Fx, Fk, FHx = circular_conv_2d_fft(image, kernel)
    SHy = torch.zeros_like(Hx)
    SHx = Hx[..., ::sf, ::sf]
    SHx_n = SHx + torch.randn_like(SHx) * sigma / 255.0
    SHy[..., ::sf, ::sf] = SHx_n
    # SHy[..., ::sf, ::sf], wiener_filter = wiener_denoise(SHx_n, sigma)

    v = torch.nn.functional.interpolate(SHx_n, scale_factor=sf, mode=mode)
    # v = torch.nn.functional.interpolate(wiener_denoise(SHx_n, sigma), scale_factor=sf, mode=mode)
    # v = torch.real(torch.fft.ifft2(torch.fft.fft2(torch.nn.functional.interpolate(SHx_n, scale_factor=sf, mode=mode)).squeeze() * mmm))

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
    jud = PSNR()
    print(f"PSNR: {jud(ans_, image)}")
    breakpoint()

    imshow(
        [
            tensor2float(image.squeeze()),
            tensor2float(Hx.squeeze()),
            tensor2float(SHx.squeeze()),
            tensor2float(SHx_n.squeeze()),
            tensor2float(SHy.squeeze()),
            tensor2float(Fx.squeeze()),
            tensor2float(FHx.squeeze()),
            tensor2float(torch.abs(torch.fft.fft2(SHx).squeeze())),
            tensor2float(torch.abs(torch.fft.fft2(SHx_n).squeeze())),
            tensor2float(torch.abs(torch.fft.fft2(SHy).squeeze())),
            # tensor2float((torch.fft.fft2(SHx_n).squeeze().real-torch.fft.fft2(SHx).squeeze().real)/torch.abs(torch.fft.fft2(SHx_n).squeeze().real-torch.fft.fft2(SHx).squeeze().real).max()),
            tensor2float(v.squeeze()),
            tensor2float(torch.abs(torch.fft.fft2(v).squeeze())),
            tensor2float(ans_.squeeze()),
            tensor2float(torch.abs(torch.fft.fft2(ans_).squeeze())),
            Fk.squeeze().unsqueeze(-1).repeat(1, 1, 3),  # type: ignore[arg-type]
            K,
            # tensor2float(torch.abs(wiener_filter).squeeze()),
            ((torch.abs(Fk) > limi).squeeze()),
            # tensor2float(torch.real(torch.fft.ifft2(torch.fft.fft2(image).squeeze() * mmm)).squeeze())
        ]
    )
