import torch
import torch.fft
from src.utils.utils_image import circular_conv_2d_fft,circular_conv_2d_conv, conv2d_via_matrix_multiplication


if __name__ == "__main__":
    from src.utils.utils_image import imshow

    image = torch.tensor([
        [1., 2., 3., 4., 99.],
        [5., 6., 7., 8., 99.],
        [9., 10., 11., 12., 77.],
        [13., 14., 15., 16., 8.],
        [17., 18., 19., 20., 44.]
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

    result_fft = circular_conv_2d_fft(image, kernel)
    result_conv = circular_conv_2d_conv(image, kernel)
    result_multi = conv2d_via_matrix_multiplication(image, kernel)
    imshow([(result_fft.squeeze()), (result_conv.squeeze()), (result_multi.squeeze()), kernel])
    print(torch.max(torch.abs(result_fft - result_multi)))
