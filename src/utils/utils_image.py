import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from pathlib import Path
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from dataclasses import dataclass, field


@dataclass
class DegradationOutput:
    H_img: torch.Tensor
    L_img: torch.Tensor
    R_img: torch.Tensor
    mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    k: torch.Tensor = field(default_factory=lambda: torch.tensor([[[1.0]]]))
    sigma: torch.Tensor = field(default_factory=lambda: torch.tensor([[[0.0]]]))
    sr: float = 0.0
    sf: int = 1

    def __post_init__(self):
        if self.mask.numel() == 0:
            self.mask = torch.ones_like(self.L_img)


class KernelSynthesizer:
    def __init__(
        self,
        trajectory_length: int = 250,
        base_kernel_size: int = 25,
    ):
        self.T = trajectory_length
        self.base_size = base_kernel_size

    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        coords = torch.arange(size) - size // 2
        x, y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def _random_trajectory(self) -> torch.Tensor:
        """Generate 3D random camera trajectory."""
        x = torch.zeros((3, self.T))
        v = torch.randn((3, self.T))
        r = torch.zeros((3, self.T))
        trv, trr = 1.0, 2 * torch.pi / self.T

        for t in range(1, self.T):
            F_rot = torch.randn(3) / (t + 1) + r[:, t - 1]
            F_trans = torch.randn(3) / (t + 1)

            r[:, t] = r[:, t - 1] + trr * F_rot
            v[:, t] = v[:, t - 1] + trv * F_trans

            R = self._rotation_matrix(r[:, t])
            st = R @ v[:, t]
            x[:, t] = x[:, t - 1] + st
        return x

    @staticmethod
    def _rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
        """Create 3D rotation matrix from Euler angles."""
        Rx = torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                [0, torch.sin(angles[0]), torch.cos(angles[0])],
            ]
        )

        Ry = torch.tensor(
            [
                [torch.cos(angles[1]), 0, torch.sin(angles[1])],
                [0, 1, 0],
                [-torch.sin(angles[1]), 0, torch.cos(angles[1])],
            ]
        )

        Rz = torch.tensor(
            [
                [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                [0, 0, 1],
            ]
        )
        return Rz @ Ry @ Rx

    def _trajectory_to_kernel(
        self, trajectory: torch.Tensor, kernel_size: int
    ) -> torch.Tensor | None:
        """Convert trajectory to blur kernel using histogram method."""
        # Project trajectory to 2D
        x_coords = trajectory[0]
        y_coords = trajectory[1]

        # Normalize coordinates to [0, kernel_size) range
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_norm = (x_coords - x_min) / (x_max - x_min + 1e-6) * (kernel_size - 1)
        y_norm = (y_coords - y_min) / (y_max - y_min + 1e-6) * (kernel_size - 1)

        # Convert to integer indices
        x_idx = torch.floor(x_norm).long()
        y_idx = torch.floor(y_norm).long()

        # Filter valid indices
        valid = (
            (x_idx >= 0) & (x_idx < kernel_size) & (y_idx >= 0) & (y_idx < kernel_size)
        )
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]

        # Create 2D histogram
        flat_indices = x_idx * kernel_size + y_idx
        counts = torch.bincount(flat_indices, minlength=kernel_size**2)
        kernel = counts.view(kernel_size, kernel_size).float()

        if kernel.sum() == 0:
            return None

        # Smooth with Gaussian
        kernel = kernel / kernel.sum()
        gauss = self.gaussian_kernel(3, 1.0)
        kernel = F.conv2d(kernel[None, None], gauss[None, None], padding=1)[0, 0]
        return kernel / kernel.sum()

    def gen_motion_kernel(
        self, target_size: int = 25, interpolation_prob: float = 0.25
    ) -> torch.Tensor:
        """Main entry point for blur kernel synthesis."""
        while True:
            trajectory = self._random_trajectory()
            kernel = self._trajectory_to_kernel(trajectory, self.base_size)
            if kernel is not None:
                break

        # Resize kernel to target size
        if kernel.shape[0] > target_size:
            kernel = kernel[:target_size, :target_size]
        else:
            pad = (target_size - kernel.shape[0]) // 2
            kernel = F.pad(kernel, (pad, pad, pad, pad))

        # Random interpolation
        if torch.rand(1).item() < interpolation_prob:
            scale = torch.randint(target_size, 5 * target_size, (2,)).tolist()
            kernel = F.interpolate(
                kernel[None, None], size=scale, mode="bilinear", align_corners=False
            )[0, 0]
            kernel = kernel[:target_size, :target_size]

        # Fallback to Gaussian if invalid
        if kernel.sum() < 0.1:
            kernel = self.gen_gaussian_kernel(target_size)

        return kernel / kernel.sum()

    # modify from https://github.com/cszn/KAIR/blob/master/utils/utils_sisr.py#L172
    def gen_gaussian_kernel(
        self,
        target_size: int = 25,
        min_var: float = 0.6,
        max_var: float = 12.0,
    ) -> torch.Tensor:
        """
        Generate a 2D Gaussian kernel with random orientation and scaling.

        Args:
            target_size: int, size of the kernel (height, width). Default is 25.
            min_var: float, minimum variance for the Gaussian distribution. Default is 0.6.
            max_var: float, maximum variance for the Gaussian distribution. Default is 12.0.
            device: Optional[torch.device], device to place the kernel on. If None, defaults to CUDA if available, else CPU.

        Returns:
            Tensor: A 2D Gaussian kernel of size target_size, normalized to sum to 1.
        """
        sf = torch.randint(1, 5, (1,)).item()
        scale_factor = torch.tensor([sf, sf], dtype=torch.float32)
        theta = torch.empty(1).uniform_(0, torch.pi)
        lambda_ = torch.empty(2).uniform_(min_var, max_var)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        Q = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta]).reshape(2, 2)
        SIGMA = Q @ torch.diag(lambda_) @ Q.T
        INV_SIGMA = torch.linalg.inv(SIGMA)
        MU = ((torch.tensor(target_size) // 2).float() - 0.5 * (scale_factor - 1)).view(
            1, 1, 2, 1
        )
        y, x = torch.meshgrid(
            torch.arange(target_size, dtype=torch.float32),
            torch.arange(target_size, dtype=torch.float32),
            indexing="ij",
        )
        coord = torch.stack([x, y], dim=-1).unsqueeze(-1)  # [H, W, 2, 1]
        delta = coord - MU
        quadratic = torch.einsum(
            "...ij,...jk,...kl->...il", delta.transpose(-1, -2), INV_SIGMA, delta
        ).squeeze((-1, -2))
        kernel = torch.exp(-0.5 * quadratic)
        return kernel / kernel.sum()

    @staticmethod
    def gen_ones_kernel():
        return torch.ones(1, 1)


def tensor2float(tensor: torch.Tensor) -> NDArray[np.float32]:
    """
    Convert a PyTorch tensor to a float32 numpy array.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The converted array as a float32 numpy array.
    """
    return tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def uint2tensor(img: NDArray[np.uint8], normalize: bool = True) -> torch.Tensor:
    """
    Convert a uint8 numpy array to a PyTorch tensor.

    Args:
        img (np.ndarray): The input image as a uint8 array with shape (H, W, 3).

    Returns:
        torch.Tensor: The converted image as a PyTorch tensor with shape (3, H, W).
    """
    return (
        torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        if normalize
        else torch.from_numpy(img.transpose(2, 0, 1)).float()
    )


def imread_uint_3(path: Path) -> NDArray[np.uint8]:
    """
    Read an image from a given path and return it as a uint8 numpy array with 3 channels.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image as a uint8 array with shape (H, W, 3).
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


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


def imshow(images: List[np.ndarray], titles: List[str] | None = None) -> None:
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
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[num_images:]:
        ax.remove()
    plt.tight_layout()
    plt.show()
