import torch
import torch.fft
from src.utils.utils_image import circular_conv_2d_fft, circular_conv_2d_conv


if __name__ == "__main__":
    from src.utils.utils_image import imshow

    image = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 99.0],
            [5.0, 6.0, 7.0, 8.0, 99.0],
            [9.0, 10.0, 11.0, 12.0, 77.0],
            [13.0, 14.0, 15.0, 16.0, 8.0],
            [17.0, 18.0, 19.0, 20.0, 44.0],
        ]
    )
    # kernel = torch.tensor([
    #     [1., 3., 9., 10.],
    #     [0., 1., 3., 4.],
    #     [10., 4., 8., 9.],
    #     [11., 12., 13., 14.],
    # ])
    kernel = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
            ],
            [
                4.0,
                5.0,
                6.0,
            ],
            [
                7.0,
                8.0,
                9.0,
            ],
        ]
    )

    result_fft = circular_conv_2d_fft(image, kernel)
    result_conv = circular_conv_2d_conv(image, kernel)
    # result_multi = conv2d_via_matrix_multiplication(image, kernel)
    imshow([(result_fft.squeeze()), (result_conv.squeeze())])
    print(torch.max(torch.abs(result_fft - result_conv)))
