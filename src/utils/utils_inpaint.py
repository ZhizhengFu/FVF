import cv2
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
from numpy.typing import NDArray
from .utils_img import imshow, imread_uint


def create_inpaint_mask(shape: Tuple[int, ...], ratio: float) -> NDArray[np.uint8]:
    """
    Creates a binary mask of the specified shape with a given ratio of ones.

    Args:
        shape (Tuple[int, ...]): The shape of the mask to be created.
        ratio (float): The ratio of the mask that should be ones (between 0 and 1).

    Returns:
        NDArray[np.uint8]: A binary mask with values 0 or 1.

    Raises:
        ValueError: If the ratio is not between 0 and 1.
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("Masking ratio must be between 0 and 1.")
    return (np.random.rand(*shape) < ratio).astype(np.uint8)


def apply_inpaint_mask(
    image: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """
    Applies a binary mask to an image, masking out the pixels where the mask is zero.

    Args:
        image (NDArray[np.uint8]): The input image to be masked.
        mask (NDArray[np.uint8]): The binary mask to apply to the image.

    Returns:
        NDArray[np.uint8]: The masked image.

    Raises:
        ValueError: If the image and mask shapes do not match.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask shapes must match.")
    return image * mask[..., np.newaxis]


def inpaint_image(
    image: NDArray[np.uint8], mask: NDArray[np.uint8], method: str = "NS"
) -> NDArray[np.uint8]:
    """
    Inpaints the masked regions of an image using the specified inpainting method.

    Args:
        image (NDArray[np.uint8]): The input image with masked regions.
        mask (NDArray[np.uint8]): The binary mask indicating the regions to inpaint.
        method (str, optional): The inpainting method to use. Options are 'NS' (Navier-Stokes)
                                or 'TELEA' (Telea's method). Defaults to "NS".

    Returns:
        NDArray[np.uint8]: The inpainted image.

    Raises:
        ValueError: If an invalid inpainting method is provided.
    """
    method_map = {"NS": cv2.INPAINT_NS, "TELEA": cv2.INPAINT_TELEA}
    if method not in method_map:
        raise ValueError("Invalid inpainting method. Use 'NS' or 'TELEA'.")

    inpainted = cv2.inpaint(
        image,
        (1 - mask),
        inpaintRadius=3,
        flags=method_map[method],
    )
    return inpainted.astype(np.uint8)


def inpaint_process(
    image: NDArray[np.uint8], ratio: float = 0.2, method: str = "NS"
) -> Dict[str, NDArray[np.uint8]]:
    """
    Processes an image by creating a mask, applying it, and then inpainting the masked regions.

    Args:
        image (NDArray[np.uint8]): The input image to process.
        ratio (float, optional): The ratio of the image to mask out. Defaults to 0.2.
        method (str, optional): The inpainting method to use. Defaults to "NS".

    Returns:
        Dict[str, NDArray[np.uint8]]: A dictionary containing the masked image,
                                      inpainted image, and the mask used.
    """
    mask = create_inpaint_mask(image.shape[:2], ratio)
    masked_image = apply_inpaint_mask(image, mask)
    inpainted_image = inpaint_image(masked_image, mask, method)

    mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

    return {
        "masked_image": masked_image,
        "inpaint_image": inpainted_image,
        "mask": mask,
    }


def main():
    img_path = Path("src/utils/test.bmp")
    image = imread_uint(img_path)
    masked_image, inpainted_image, mask = inpaint_process(
        image, ratio=0.2, method="NS"
    ).values()
    imshow([masked_image, inpainted_image], ["Masked Image", "NS Inpainting"])


if __name__ == "__main__":
    main()
