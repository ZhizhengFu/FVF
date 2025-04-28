import torch
import torch.fft

def splits_and_mean(a, sf):
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return torch.mean(b, dim=-1)

def conv2d_via_matrix_multiplication(image, kernel):
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    h, w = image.shape[-2:]
    kh, kw = kernel.shape[-2:]

    H = construct_circulant_matrix(kernel.squeeze(), h, w)

    flat_image = image.squeeze().flatten()
    result = H @ flat_image

    return result.reshape(h, w), H, flat_image

def construct_circulant_matrix(kernel, h, w):
    kh, kw = kernel.shape
    H = torch.zeros(h*w, h*w)

    for i in range(h):
        for j in range(w):
            row_idx = i*w + j
            for ki in range(kh):
                for kj in range(kw):
                    pos_i = (i + ki - kh + 1) % h
                    pos_j = (j + kj - kw + 1) % w
                    col_idx = pos_i*w + pos_j
                    H[row_idx, col_idx] = kernel[ki, kj]
    return H

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

    return torch.real(result)

if __name__ == "__main__":
    from src.utils.utils_image import imshow, imread_uint_3, uint2tensor, tensor2float, KernelSynthesizer
    from pathlib import Path

    image = uint2tensor(imread_uint_3(Path("src/utils/test.png"))).unsqueeze(0)
    kernel = KernelSynthesizer().gen_motion_kernel()

    sf = 2
    result_fft = circular_conv_2d_fft(image, kernel)
    y = torch.zeros_like(result_fft)
    result_fft = result_fft[...,::sf,::sf]
    y[...,::sf,::sf] = result_fft
    v = torch.nn.functional.interpolate(result_fft,scale_factor=sf, mode="nearest")

    alpha = 0.000005
    t_kernel = torch.flip(kernel, [-1, -2])

    FK = torch.fft.fft2(t_kernel.to(torch.float64), s=(image.size(-2), image.size(-1))).unsqueeze(0).unsqueeze(0)

    v, alpha = v.to(torch.float64), alpha
    FCK, F2K = torch.conj(FK), torch.abs(FK) ** 2
    FCKFSHy = FCK * torch.fft.fft2(y)
    FR = FCKFSHy + torch.fft.fft2(alpha * v)
    _FKFR_, _F2K_ = splits_and_mean(FK * FR, sf), splits_and_mean(F2K, sf)
    _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
    FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(1, 1, sf, sf)
    FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / alpha
    ans_ = torch.real(torch.fft.ifft2(FX))

    imshow([tensor2float(image.squeeze()), tensor2float(v.squeeze()), tensor2float(ans_.squeeze()), kernel])
