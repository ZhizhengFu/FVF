import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from numpy.typing import NDArray
from .utils_image import imshow, imread_uint


def mosaic_CFA_Bayer_pipeline(
    image: NDArray[np.uint8], pattern: str = "RGGB", method: str = "EA"
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Combines Bayer CFA (Color Filter Array) mosaicing and demosaicing into a single pipeline.

    Args:
        image (np.ndarray): The input image to process.
        pattern (str, optional): The Bayer pattern to use (e.g., 'RGGB', 'BGGR'). Defaults to "RGGB".
        method (str, optional): The demosaicing method to use ('EA' for Edge-Aware, 'VNG' for Variable Number of Gradients). Defaults to "EA".

    Returns:
        Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]: A tuple containing the mosaiced image, demosaicked image, and mask.
    """
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(
            "Invalid Bayer pattern. Use 'RGGB', 'BGGR', 'GRBG', or 'GBRG'."
        )

    if method not in {"EA", "VNG"}:
        raise ValueError("Invalid demosaicing method. Use 'EA' or 'VNG'.")

    mask = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    channels = "RGB"

    for channel, (y, x) in zip(pattern, positions):
        idx = channels.index(channel)
        mask[y::2, x::2, idx] = True

    mosaiced_image = image * mask
    CFA = mosaiced_image.sum(axis=2).astype(np.uint8)

    method_map = {
        "EA": cv2.COLOR_BAYER_BG2RGB_EA,
        "VNG": cv2.COLOR_BAYER_BG2RGB_VNG,
    }

    demosaicked_image = cv2.cvtColor(CFA, method_map[method])

    return mosaiced_image, demosaicked_image, mask * 255


def main():
    img_path = Path("src/utils/test.bmp")
    image = imread_uint(img_path)
    mosaiced_image, demosaicked_image, mask = mosaic_CFA_Bayer_pipeline(
        image, pattern="RGGB", method="EA"
    )
    imshow(
        [mosaiced_image, demosaicked_image, mask],
        ["Mosaiced Image", "Demosaicked Image", "Bayer Mask"],
    )


if __name__ == "__main__":
    main()
