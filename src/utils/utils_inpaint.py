import cv2
import torch
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from .utils_image import uint2tensor, DegradationOutput


def inpaint_pipeline(
    H_img: NDArray[np.uint8],
    sr: float,
    device: torch.device,
    method: str = "NS",
) -> DegradationOutput:
    if not (0.0 <= sr <= 1.0):
        raise ValueError("Masking ratio must be between 0 and 1.")
    if method not in {"NS", "TELEA"}:
        raise ValueError("Invalid inpainting method. Use 'NS' or 'TELEA'.")
    mask = (np.random.rand(*H_img.shape[:2]) < sr).astype(np.uint8)
    L_img = H_img * mask[..., np.newaxis]
    method_map = {"NS": cv2.INPAINT_NS, "TELEA": cv2.INPAINT_TELEA}
    R_img = cv2.inpaint(
        L_img,
        ~mask & 1,
        inpaintRadius=3,
        flags=method_map[method],
    ).astype(np.uint8)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

    H_img_tensor = uint2tensor(H_img).to(device)
    L_img_tensor = uint2tensor(L_img).to(device)
    R_img_tensor = uint2tensor(R_img).to(device)
    mask_tensor = uint2tensor(mask, False).to(device)

    return DegradationOutput(
        H_img=H_img_tensor,
        L_img=L_img_tensor,
        R_img=R_img_tensor,
        mask=mask_tensor,
        sr=sr,
    )


def main():
    from .utils_image import imread_uint_3, imshow, tensor2float

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = imread_uint_3(Path("src/utils/test.png"))
    inpaint_return = inpaint_pipeline(image, 0.2, device, method="NS")
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
