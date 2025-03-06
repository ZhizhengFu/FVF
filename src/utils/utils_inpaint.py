import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from numpy.typing import NDArray
from .utils_img import imshow, imread_uint


def inpaint_pipeline(
    image: NDArray[np.uint8], ratio: float = 0.2, method: str = "NS"
) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """
    Combines mask creation, application, and inpainting into a single streamlined pipeline.

    Args:
        image (NDArray[np.uint8]): The input image to process.
        ratio (float, optional): The ratio of the image to mask out. Defaults to 0.2.
        method (str, optional): The inpainting method to use ('NS' for Navier-Stokes, 'TELEA' for Telea's method). Defaults to "NS".

    Returns:
        Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]: A tuple containing the masked image, inpainted image, and mask.
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("Masking ratio must be between 0 and 1.")

    if method not in {"NS", "TELEA"}:
        raise ValueError("Invalid inpainting method. Use 'NS' or 'TELEA'.")

    mask = (np.random.rand(*image.shape[:2]) < ratio).astype(np.uint8)
    masked_image = image * mask[..., np.newaxis]
    method_map = {"NS": cv2.INPAINT_NS, "TELEA": cv2.INPAINT_TELEA}

    inpainting_image = cv2.inpaint(
        masked_image,
        ~mask & 1,
        inpaintRadius=3,
        flags=method_map[method],
    ).astype(np.uint8)

    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

    return masked_image, inpainting_image, mask * 255


def main():
    img_path = Path("src/utils/test.bmp")
    image = imread_uint(img_path)
    masked_image, inpainting_image, mask = inpaint_pipeline(
        image, ratio=0.2, method="NS"
    )
    imshow(
        [masked_image, inpainting_image, mask],
        ["Masked Image", "Inpainting Image", "Mask"],
    )


if __name__ == "__main__":
    main()
