import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Literal
from numpy.typing import NDArray
from .utils_image import uint2tensor, DegradationOutput


def mosaic_CFA_Bayer_pipeline(
    H_img: NDArray[np.uint8],
    pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] = "RGGB",
    method: Literal["EA", "VNG"] = "EA",
    sigma: float | None = None,
    sigma_max: float = 25,
) -> DegradationOutput:
    mask = np.zeros((*H_img.shape[:2], 3), dtype=np.uint8)
    _sigma = sigma if sigma is not None else np.random.uniform(0, sigma_max / 255.0)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    channels = "RGB"
    for channel, (y, x) in zip(pattern, positions):
        idx = channels.index(channel)
        mask[y::2, x::2, idx] = True
    L_img = H_img * mask
    CFA = L_img.sum(axis=2).astype(np.uint8)
    CFA_noisy = CFA.astype(np.float32) + _sigma * 255 * np.random.randn(*CFA.shape)
    CFA_noisy = np.clip(CFA_noisy, 0, 255).astype(np.uint8)
    R_img = cv2.cvtColor(CFA_noisy, getattr(cv2, f"COLOR_BAYER_BG2RGB_{method}"))

    H_img_tensor = uint2tensor(H_img)
    L_img_tensor = uint2tensor(L_img)
    R_img_tensor = uint2tensor(R_img)
    mask_tensor = uint2tensor(mask, False)

    return DegradationOutput(
        H_img=H_img_tensor,
        L_img=L_img_tensor,
        R_img=R_img_tensor,
        mask=mask_tensor,
        sigma=torch.tensor([_sigma]).view([1, 1, 1]),
        type=2,
    )


def main():
    from .utils_image import imread_uint_3, imshow, tensor2float

    image = imread_uint_3(Path("src/utils/test.png"))
    mosaic_return = mosaic_CFA_Bayer_pipeline(image, pattern="RGGB", method="EA")
    imshow(
        [
            tensor2float(mosaic_return.H_img),
            tensor2float(mosaic_return.L_img),
            tensor2float(mosaic_return.R_img),
            tensor2float(mosaic_return.mask),
        ]
    )


if __name__ == "__main__":
    main()
