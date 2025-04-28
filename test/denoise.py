import torch
import torch.fft
import pywt
import numpy as np
from src.utils.utils_image import (
    imshow,
    imread_uint_3,
    uint2tensor,
    tensor2float,
    KernelSynthesizer,
)
from pathlib import Path

# def wiener_filter_fft(noisy_image, sigma, epsilon=1e-8):
#     """频域维纳滤波去噪"""
#     F_noisy = torch.fft.fft2(noisy_image)
#     power_noisy = torch.abs(F_noisy) ** 2
#     power_signal = torch.clamp_min(power_noisy - sigma ** 2, epsilon)  # 估计信号功率谱
#     wiener_filter = power_signal / (power_signal + sigma ** 2)  # 维纳滤波器
#     F_denoised = F_noisy * wiener_filter
#     return torch.real(torch.fft.ifft2(F_denoised))
def fft_denoise(signal, sigma):
    H, W = signal.shape[-2], signal.shape[-1]
    signal_fft = torch.fft.fft2(signal)
    signal_fft_shifted = torch.fft.fftshift(signal_fft, dim=(-2, -1))

    # 计算噪声功率谱 (考虑FFT的幅度平方特性)
    Snn = (sigma / 255.0) ** 2 * H * W

    # 维纳滤波参数计算
    magnitude_sq = torch.abs(signal_fft_shifted) ** 2
    Sxx_estimate = torch.clamp(magnitude_sq - Snn, min=0)
    wiener_filter = Sxx_estimate / (Sxx_estimate + Snn + 1e-6)

    # 应用滤波器并逆变换
    denoised_fft = torch.fft.ifftshift(signal_fft_shifted * wiener_filter, dim=(-2, -1))
    return torch.real(torch.fft.ifft2(denoised_fft))


def wavelet_denoise(input_img, noise_sigma, wavelet='db4', level=3):
    """小波阈值去噪"""
    np_img = input_img.squeeze().cpu().numpy()
    coeffs = pywt.wavedec2(np_img, wavelet, level=level)

    # 阈值估计（BayesShrink）
    threshold = noise_sigma * np.sqrt(2 * np.log(np_img.size))

    # 各层高频系数阈值处理
    coeffs_thresh = [coeffs[0]] + [
        (pywt.threshold(cH, threshold, mode='soft'),
            pywt.threshold(cV, threshold, mode='soft'),
            pywt.threshold(cD, threshold, mode='soft'))
        for cH, cV, cD in coeffs[1:]
    ]

    restored_np = pywt.waverec2(coeffs_thresh, wavelet)
    return torch.from_numpy(restored_np).unsqueeze(0).to(input_img.device)

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

    return torch.real(result), torch.real(fft_image), torch.real(fft_kernel), torch.real(product)

def reconstruct_image(Hx, SHx_n, kernel, sf, alpha, mode="nearest"):
    SHy = torch.zeros_like(Hx)
    SHy[..., ::sf, ::sf] = SHx_n

    v = torch.nn.functional.interpolate(SHx_n, scale_factor=sf, mode=mode)
    K = torch.flip(kernel, [-1, -2])
    FK = (
        torch.fft.fft2(K.to(torch.float64), s=(Hx.size(-2), Hx.size(-1)))
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

    return ans_

if __name__ == "__main__":
    # 加载图像和参数设置
    image = uint2tensor(imread_uint_3(Path("src/utils/test.png"))).unsqueeze(0)
    kernel = KernelSynthesizer().gen_motion_kernel()
    # kernel = torch.tensor([[1]])
    sf = 2
    sigma = 8
    mode = "nearest"
    alpha = 1e-1

    # 模拟降质过程
    Hx, Fx, Fk, FHx = circular_conv_2d_fft(image, kernel)
    SHx = Hx[..., ::sf, ::sf]
    SHx_n = SHx + torch.randn_like(SHx) * sigma / 255.

    # 不去噪的重建
    recon_no_denoise = reconstruct_image(Hx, SHx_n, kernel, sf, alpha, mode)

    # 维纳滤波去噪后的重建
    SHx_n_wiener = fft_denoise(SHx_n, sigma/255.)
    recon_wiener = reconstruct_image(Hx, SHx_n_wiener, kernel, sf, alpha, mode)

    # 小波去噪后的重建
    SHx_n_wavelet = wavelet_denoise(SHx_n, sigma/255.)
    recon_wavelet = reconstruct_image(Hx, SHx_n_wavelet, kernel, sf, alpha, mode)

    # 显示结果对比
    imshow(
        [
            tensor2float(image.squeeze()),  # 原始图像
            tensor2float(Hx.squeeze()),    # 模糊后的图像
            tensor2float(SHx_n.squeeze()), # 降采样+加噪的图像
            tensor2float(SHx_n_wiener.squeeze()),  # 维纳滤波去噪结果
            tensor2float(SHx_n_wavelet.squeeze()), # 小波去噪结果
            tensor2float(recon_no_denoise.squeeze()),  # 不去噪的重建结果
            tensor2float(recon_wiener.squeeze()),      # 维纳滤波后的重建结果
            tensor2float(recon_wavelet.squeeze()),      # 小波去噪后的重建结果
        ],
        titles=[
            "Original Image",
            "Blurred Image",
            "Downsampled+Noisy",
            "Wiener Denoised",
            "Wavelet Denoised",
            "Recon (No Denoise)",
            "Recon (Wiener)",
            "Recon (Wavelet)",
        ]
    )
