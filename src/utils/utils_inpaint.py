import cv2
import torch
import numpy as np
from typing import Tuple
from pathlib import Path
from numpy.typing import NDArray
from .utils import DegradationType
from .utils_image import uint2tensor


def inpaint_pipeline(
    H_img: NDArray[np.uint8],
    device: torch.device,
    ratio: Tuple[float, float] = (0.2, 0.5),
    method: str = "NS",
) -> Tuple[
    DegradationType,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Combines mask creation, application, and inpainting into a single streamlined pipeline.

    Args:
        H_img (np.ndarray): The input image as a NumPy array (H x W x C).
        ratio (float, optional): The ratio of the image to mask out. Defaults to 0.2.
        method (str, optional): The inpainting method to use ('NS' for Navier-Stokes, 'TELEA' for Telea's method). Defaults to "NS".
        crop_size (Optional[Tuple[int, int]], optional): If provided, crops the image to the specified size (H, W). Defaults to None.
        random_augment (bool, optional): If True, applies random augmentation (e.g., flipping). Defaults to True.

    Returns:
        Tuple[Degradation_Type, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the original image, masked image, inpainted image, and mask as PyTorch tensors.
    """
    if not (0.0 <= ratio[0] <= ratio[1] <= 1.0):
        raise ValueError("Masking ratio must be between 0 and 1.")
    if method not in {"NS", "TELEA"}:
        raise ValueError("Invalid inpainting method. Use 'NS' or 'TELEA'.")

    selected_ratio = np.random.uniform(ratio[0], ratio[1])
    mask = (np.random.rand(*H_img.shape[:2]) < selected_ratio).astype(np.uint8)
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
    sr_tensor = torch.tensor(selected_ratio).view(1, 1, 1).to(device)

    return (
        DegradationType.INPAINTING,
        H_img_tensor,
        L_img_tensor,
        R_img_tensor,
        mask_tensor,
        sr_tensor,
    )


def main():
    from .utils_image import imread_uint_3, imshow, tensor2float

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = imread_uint_3(Path("src/utils/test.png"))
    type, H_img, L_img, R_img, mask, sr = inpaint_pipeline(
        image, device, ratio=(0.2, 0.5), method="NS"
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
