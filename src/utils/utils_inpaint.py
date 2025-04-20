import cv2
import torch
import numpy as np
from typing import Literal, Annotated
from numpy.typing import NDArray
from .utils_image import uint2tensor, DegradationOutput


def inpaint_pipeline(
    H_img: NDArray[np.uint8],
    sr: Annotated[float, "Value must be between 0.0 and 1.0"],
    method: Literal["NS", "TELEA"] = "NS",
    sigma: int | None = None,
    sigma_max: float = 25,
) -> DegradationOutput:
    mask = (np.random.rand(*H_img.shape[:2]) < sr).astype(np.uint8)
    _sigma = sigma if sigma is not None else np.random.uniform(0, sigma_max)
    L_img = (
        H_img * mask[..., np.newaxis] + _sigma * np.random.randn(*H_img.shape)
    ).astype(np.uint8)
    R_img = cv2.inpaint(
        L_img,
        ~mask & 1,
        inpaintRadius=3,
        flags=getattr(cv2, f"INPAINT_{method}"),
    ).astype(np.uint8)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

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
        sr=sr,
        type=3,
    )


def main():
    from .utils_image import imread_uint_3, imshow, tensor2float
    from pathlib import Path

    image = imread_uint_3(Path("src/utils/test.png"))
    inpaint_return = inpaint_pipeline(image, 0.2, method="NS")
    imshow(
        [
            tensor2float(inpaint_return.H_img),
            tensor2float(inpaint_return.L_img),
            tensor2float(inpaint_return.R_img),
            tensor2float(inpaint_return.mask),
        ]
    )


if __name__ == "__main__":
    main()
