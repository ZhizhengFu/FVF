import torch

kernel = torch.tensor(
    [
        [1.0, 3.0, 9.0, 10.0],
        [0.0, 1.0, 3.0, 4.0],
        [10.0, 4.0, 8.0, 9.0],
        [11.0, 12.0, 13.0, 14.0],
    ]
)
fft1 = torch.fft.fft2(kernel)
fft_rows = torch.fft.fft(kernel, dim=1)
fft_2d = torch.fft.fft(fft_rows, dim=0)
print(torch.allclose(fft1, fft_2d))
