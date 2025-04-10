import torch
import torch.nn.functional as F
import random
import numpy as np
from pathlib import Path
from typing import Literal
from numpy.typing import NDArray
from .utils_image import (
    uint2tensor,
    circular_conv_2d_fft,
    DegradationOutput,
    KernelSynthesizer,
)


def sisr_pipeline(
    H_img: NDArray[np.uint8],
    sf: int,
    k_synthesizer: KernelSynthesizer = KernelSynthesizer(),
    k_size: int = 25,
    k_type: Literal["gaussian", "motion", None] = None,
    sigma: float | None = None,
    sigma_max: float = 25,
) -> DegradationOutput:
    H_img_tensor = uint2tensor(H_img)
    _sigma = (
        torch.tensor(sigma) if sigma else torch.empty(1).uniform_(0, sigma_max / 255.0)
    )
    if k_type is None:
        k_type = random.choice(["gaussian", "motion"])
    k = getattr(k_synthesizer, f"gen_{k_type}_kernel")(k_size)
    L_img_tensor = circular_conv_2d_fft(H_img_tensor.unsqueeze(0), k)[..., 0::sf, 0::sf]
    L_img_tensor, H_img_tensor = L_img_tensor.squeeze(), H_img_tensor.squeeze()
    L_img_tensor = L_img_tensor + _sigma * torch.randn_like(L_img_tensor)
    L_img_tensor = torch.clamp(L_img_tensor, 0, 1)
    R_img_tensor = F.interpolate(
        L_img_tensor.unsqueeze(0), scale_factor=sf, mode="nearest"
    ).squeeze()

    return DegradationOutput(
        H_img=H_img_tensor,
        L_img=L_img_tensor,
        R_img=R_img_tensor,
        k=k.unsqueeze(0),
        sigma=_sigma.view([1, 1, 1]),
        sf=sf,
        type=1,
    )


def main():
    from .utils_image import imshow, tensor2float, imread_uint_3

    image = imread_uint_3(Path("src/utils/test.png"))
    sisr_return = sisr_pipeline(image, 3)
    imshow(
        [
            tensor2float(sisr_return.H_img),
            tensor2float(sisr_return.L_img),
            tensor2float(sisr_return.R_img),
            tensor2float(sisr_return.k),
        ]
    )


if __name__ == "__main__":
    main()
