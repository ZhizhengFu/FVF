import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from numpy.typing import NDArray


def tensor2float(tensor: torch.Tensor) -> NDArray[np.float32]:
    """
    Convert a PyTorch tensor to a float32 numpy array.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The converted image as a float32 numpy array.
    """
    return tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)


def float2tensor(image: NDArray[np.float32]) -> torch.Tensor:
    """
    Convert a float32 numpy array to a PyTorch tensor.

    Args:
        image (np.ndarray): The input image as a float32 numpy array.

    Returns:
        torch.Tensor: The converted image as a PyTorch tensor.
    """
    return torch.from_numpy(image).permute(2, 0, 1).float()


def uint2tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    """
    Convert a uint8 numpy array to a PyTorch tensor.

    Args:
        image (np.ndarray): The input image as a uint8 numpy array.

    Returns:
        torch.Tensor: The converted image as a PyTorch tensor.
    """
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)


def imread_uint(path: Path, n_channels: int = 3) -> NDArray[np.uint8]:
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

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")

    if n_channels == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img[..., np.newaxis].astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.uint8)


def imread_float(path: Path, n_channels: int = 3) -> NDArray[np.float32]:
    """
    Read an image from a given path and return it as a float32 numpy array.

    Args:
        path (str): Path to the image file.
        n_channels (int): Number of channels in the output image (1 for grayscale, 3 for RGB).

    Returns:
        np.ndarray: The loaded image as a float32 array with shape (H, W, C).
    """
    return imread_uint(path, n_channels).astype(np.float32) / 255.0


def imshow(images: List, titles: Optional[List[str]] = None) -> None:
    """
    Display a list of images with optional titles using automatic layout.

    Args:
        images (List[np.ndarray]): List of images to display.
        titles (Optional[List[str]]): Optional titles for each image.
    """
    num_images = len(images)
    if titles is None:
        titles = [""] * num_images
    elif len(titles) != num_images:
        raise ValueError("Number of images and titles must match.")

    cols = min(num_images, 5)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for img in images:
        if isinstance(img, torch.Tensor):
            img = tensor2float(img)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[num_images:]:
        ax.remove()

    plt.tight_layout()
    plt.show()
