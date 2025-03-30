import torch
import torch.nn.functional as F
import random
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from .utils_image import uint2tensor, DegradationOutput, KernelSynthesizer


def sisr_pipeline(
    H_img: NDArray[np.uint8],
    sf: int,
    device: torch.device,
    k_synthesizer: KernelSynthesizer = KernelSynthesizer(),
    k_size: int = 25,
    sigma_max: float = 25,
    remove_random: bool = False,
) -> DegradationOutput:
    H_img_tensor = uint2tensor(H_img).to(device)
    sigma = (
        (
            torch.tensor(0.0, device=device)
            if torch.randint(0, 9, (1,)).item() == 1
            else torch.empty(1, device=device).uniform_(0, sigma_max / 255.0)
        )
        if not remove_random
        else torch.tensor(0.0, device=device)
    )
    kernel_generator = (
        random.choice(
            [k_synthesizer.gen_gaussian_kernel, k_synthesizer.gen_motion_kernel]
        )
        if not remove_random
        else k_synthesizer.gen_gaussian_kernel
    )
    k = (
        kernel_generator(k_size)
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(H_img_tensor.shape[0], -1, -1, -1)
    )
    pad_size = k.shape[-1] // 2
    L_img_tensor = F.pad(
        H_img_tensor, (pad_size, pad_size, pad_size, pad_size), mode="circular"
    ).unsqueeze(0)
    L_img_tensor = torch.conv2d(L_img_tensor, k, groups=L_img_tensor.shape[1])[
        ..., 0::sf, 0::sf
    ].squeeze()
    L_img_tensor = L_img_tensor + sigma * torch.randn_like(L_img_tensor)
    R_img_tensor = F.interpolate(
        L_img_tensor.unsqueeze(0), scale_factor=sf, mode="nearest"
    ).squeeze()
    # mask = torch.zeros_like(R_img_tensor, device=device)
    # mask[...,0::sf,0::sf] = 1

    return DegradationOutput(
        H_img=H_img_tensor,
        L_img=L_img_tensor,
        R_img=R_img_tensor,
        # mask=mask,
        k=k[0],
        sigma=sigma.view([1, 1, 1]),
        sf=sf,
        sr=1.0 / sf,
    )


def main():
    from .utils_image import imshow, tensor2float, imread_uint_3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = imread_uint_3(Path("src/utils/test.png"))
    sisr_return = sisr_pipeline(image, 3, device)
    imshow(
        [
            tensor2float(sisr_return.H_img),
            tensor2float(sisr_return.L_img),
            tensor2float(sisr_return.R_img),
            tensor2float(sisr_return.k[0].squeeze().unsqueeze(0)),
        ]
    )


if __name__ == "__main__":
    main()
