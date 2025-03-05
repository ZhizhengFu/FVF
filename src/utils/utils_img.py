import cv2
import numpy as np

def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    """
    Read an image from a given path and return it as a uint8 numpy array.

    Args:
        path (str): Path to the image file.
        n_channels (int): Number of channels in the output image (1 for grayscale, 3 for RGB).

    Returns:
        np.ndarray: The loaded image as a uint8 array with shape (H, W, C).
    """
    if n_channels not in (1, 3):
        raise ValueError("n_channels must be either 1 (grayscale) or 3 (RGB).")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    if n_channels == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img[..., np.newaxis]

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.uint8)

def imread_float(path: str, n_channels: int = 3) -> np.ndarray:
    """
    Read an image from a given path and return it as a float32 numpy array.

    Args:
        path (str): Path to the image file.
        n_channels (int): Number of channels in the output image (1 for grayscale, 3 for RGB).

    Returns:
        np.ndarray: The loaded image as a float32 array with shape (H, W, C).
    """
    return imread_uint(path, n_channels).astype(np.float32) / 255.0
