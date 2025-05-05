import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import Literal
from numpy.typing import NDArray
from .utils_image import (
    uint2tensor,
    wiener_denoiser,
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
    sigma: int | None = None,
    sigma_max: float = 25,
) -> DegradationOutput:
    H_img_tensor = uint2tensor(H_img)
    _sigma = (
        torch.tensor(sigma / 255.0)
        if sigma is not None
        else torch.empty(1).uniform_(0, sigma_max / 255.0)
    )
    if k_type is None:
        k_type = random.choice(["gaussian", "motion"])
    k = getattr(k_synthesizer, f"gen_{k_type}_kernel")(k_size)
    kh, kw = k.shape[-2:]
    pad_h = kh - 1
    pad_w = kw - 1
    Hx = circular_conv_2d_fft(H_img_tensor.unsqueeze(0), k)
    L_img_tensor = Hx + _sigma * torch.randn_like(Hx)
    R_img_tensor = torch.roll(
        L_img_tensor, shifts=[-(pad_h // 2), -(pad_w // 2)], dims=(-2, -1)
    )
    L_img_tensor = L_img_tensor[..., ::sf, ::sf].squeeze()
    R_img_tensor = R_img_tensor[..., ::sf, ::sf]
    R_img_tensor = F.interpolate(
        wiener_denoiser(R_img_tensor, _sigma), scale_factor=sf, mode="nearest"
    ).squeeze()

    return DegradationOutput(
        H_img=H_img_tensor,
        L_img=L_img_tensor,
        R_img=R_img_tensor,
        k=k.unsqueeze(0),
        sigma=(_sigma * 255).view([1, 1, 1]),
        sf=sf,
        type=1,
    )


def main():
    from .utils_image import imshow, tensor2float, imread_uint_3
    from pathlib import Path

    image = imread_uint_3(Path("src/utils/test.png"))
    sisr_return = sisr_pipeline(image, 4, sigma=0, k_type="gaussian")
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
