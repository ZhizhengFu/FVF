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

    return torch.real(result), fft_kernel

if __name__ == "__main__":
    from src.utils.utils_image import imshow
    image = torch.tensor([
        [1., 2., 3., 4., 99., 8],
        [5., 6., 7., 8., 99., 9],
        [9., 10., 11., 12., 77., 1],
        [13., 14., 15., 16., 8., 2],
        [17., 18., 19., 20., 44., 3],
        [21., 22., 23., 24., 55., 4]
    ])
    # kernel = torch.tensor([
    #     [1., 3., 9., 10.],
    #     [0., 1., 3., 4.],
    #     [10., 4., 8., 9.],
    #     [11., 12., 13., 14.],
    # ])
    kernel = torch.tensor([
        [1., 2., 3.,],
        [4., 5., 6.,],
        [7., 8., 9.,],
    ])
    S = torch.zeros(36, 36)
    for i in [0,2,4,12,14,16,24,26,28]:
        S[i][i]=1
    h, w = image.shape
    result_fft, FK = circular_conv_2d_fft(image, kernel)
    result_multi, H, flat_image = conv2d_via_matrix_multiplication(image, kernel)

    y = S @ H @ flat_image

    v = torch.nn.functional.interpolate((H @ flat_image).reshape(6,6)[::2,::2].unsqueeze(0).unsqueeze(0),scale_factor=2, mode="nearest").squeeze().flatten(start_dim=0)
    alpha = 0.05
    left =H.H@S@H+alpha*torch.eye(36)
    right=H.H@S.H@y+alpha*v
    x_ans =torch.linalg.solve(left,right)

    FCK, F2K = torch.conj(FK), torch.abs(FK) ** 2
    FCKFSHy = FCK * torch.fft.fft2(y.reshape(6,6))
    FR = FCKFSHy + torch.fft.fft2(alpha * v.reshape(6,6))
    _FKFR_, _F2K_ = splits_and_mean(FK * FR, 2), splits_and_mean(F2K, 2)
    _FKFR_FMdiv_FK2_FM = _FKFR_ / (_F2K_ + alpha)
    FCK_FKFR_FMdiv_FK2_FM = FCK * _FKFR_FMdiv_FK2_FM.repeat(1, 1, 2, 2)
    FX = (FR - FCK_FKFR_FMdiv_FK2_FM) / alpha
    ans_ = torch.real(torch.fft.ifft2(FX))

    breakpoint()
    # ans_ == x_ans?
    imshow([(result_fft.squeeze()), (result_conv.squeeze()), (result_multi.squeeze()), kernel, H])
