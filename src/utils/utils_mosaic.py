import cv2
import torch
import numpy as np
from typing import Tuple
from pathlib import Path
from numpy.typing import NDArray
from .utils import DegradationType
from .utils_image import uint2tensor


def mosaic_CFA_Bayer_pipeline(
    H_img: NDArray[np.uint8],
    device: torch.device,
    pattern: str = "RGGB",
    method: str = "EA",
) -> Tuple[DegradationType, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines Bayer CFA (Color Filter Array) mosaicing and demosaicing into a single pipeline.

    Args:
        image (np.ndarray): The input image to process.
        pattern (str, optional): The Bayer pattern to use (e.g., 'RGGB', 'BGGR'). Defaults to "RGGB".
        method (str, optional): The demosaicing method to use ('EA' for Edge-Aware, 'VNG' for Variable Number of Gradients). Defaults to "EA".
        crop_size (int, optional): The size of the patch for the Edge-Aware method. Defaults to 96.

    Returns:
        Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]: A tuple containing the mosaiced image, demosaicked image, and mask.
    """
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(
            "Invalid Bayer pattern. Use 'RGGB', 'BGGR', 'GRBG', or 'GBRG'."
        )
    if method not in {"EA", "VNG"}:
        raise ValueError("Invalid demosaicing method. Use 'EA' or 'VNG'.")

    mask = np.zeros((*H_img.shape[:2], 3), dtype=np.uint8)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    channels = "RGB"
    for channel, (y, x) in zip(pattern, positions):
        idx = channels.index(channel)
        mask[y::2, x::2, idx] = True
    L_img = H_img * mask
    CFA = L_img.sum(axis=2).astype(np.uint8)
    method_map = {
        "EA": cv2.COLOR_BAYER_BG2RGB_EA,
        "VNG": cv2.COLOR_BAYER_BG2RGB_VNG,
    }
    R_img = cv2.cvtColor(CFA, method_map[method])
    H_img_tensor = uint2tensor(H_img).to(device)
    L_img_tensor = uint2tensor(L_img).to(device)
    R_img_tensor = uint2tensor(R_img).to(device)
    mask_tensor = uint2tensor(mask, False).to(device)
    return (
        DegradationType.MOSAIC,
        H_img_tensor,
        L_img_tensor,
        R_img_tensor,
        mask_tensor,
    )


def main():
    from .utils_image import imread_uint_3, imshow, tensor2float

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = imread_uint_3(Path("src/utils/test.png"))
    type, H_img, L_img, R_img, mask = mosaic_CFA_Bayer_pipeline(
        image,
        device,
        pattern="RGGB",
        method="EA",
    )
    imshow(
        [
            tensor2float(H_img),
            tensor2float(L_img),
            tensor2float(R_img),
            tensor2float(mask),
        ]
    )
    print(H_img.shape, L_img.shape, R_img.shape, mask.shape)


if __name__ == "__main__":
    main()
