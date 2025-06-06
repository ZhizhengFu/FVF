import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from pathlib import Path
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from dataclasses import dataclass, field


def wiener_denoiser(SHx_n, sigma):
    sigma_norm = sigma / 255.0
    noise_power = sigma_norm**2
    signal_fft = torch.fft.fft2(SHx_n)
    signal_power = torch.abs(signal_fft) ** 2 / torch.numel(SHx_n)
    wiener_filter = signal_power / (signal_power + noise_power)
    denoised_fft = signal_fft * wiener_filter
    denoised = torch.fft.ifft2(denoised_fft).real
    return denoised


def circular_conv_2d_via_matrix_multiplication(image, kernel):
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    h, w = image.shape[-2:]
    kh, kw = kernel.shape[-2:]

    H = _construct_circulant_matrix(kernel.squeeze(), h, w)
    flat_image = image.squeeze().flatten()
    result = H @ flat_image

    return result.reshape(h, w), H, flat_image


def _construct_circulant_matrix(kernel, h, w):
    kh, kw = kernel.shape
    H = torch.zeros(h * w, h * w)

    for i in range(h):
        for j in range(w):
            row_idx = i * w + j
            for ki in range(kh):
                for kj in range(kw):
                    pos_i = (i + ki - kh + 1) % h
                    pos_j = (j + kj - kw + 1) % w
                    col_idx = pos_i * w + pos_j
                    H[row_idx, col_idx] = kernel[ki, kj]
    return H


def circular_conv_2d_fft(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    batch_size, channels, h, w = image.shape
    kh, kw = kernel.shape[-2:]

    kernel = torch.flip(kernel, [-1, -2])
    fft_image = torch.fft.fft2(image)
    fft_kernel = torch.fft.fft2(kernel, s=(h, w))
    product = fft_image * fft_kernel
    result = torch.fft.ifft2(product)

    return torch.real(result)


def circular_conv_2d_conv(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    batch_size, channels, h, w = image.shape
    kernel = kernel.repeat(channels, 1, 1, 1)
    kh, kw = kernel.shape[-2:]
    pad_h = kh - 1
    pad_w = kw - 1
    image_pad = F.pad(image, (pad_w, 0, pad_h, 0), mode="circular")
    result = F.conv2d(image_pad, kernel, padding=0, groups=channels)
    return result


class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 1]"""

    def __init__(self, border=0):
        super().__init__()
        self.border = border

    def forward(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")

        if self.border > 0:
            img1 = img1[..., self.border : -self.border, self.border : -self.border]
            img2 = img2[..., self.border : -self.border, self.border : -self.border]

        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = -10.0 * torch.log10(mse)
        psnr = torch.clamp(psnr, min=0.0, max=float("inf"))
        return torch.mean(psnr)  # return average PSNR over batch


class SSIM(nn.Module):
    """Structural Similarity Index Measure
    img1 and img2 have range [0, 1]"""

    def __init__(self, border=0, window_size=11, size_average=True):
        super().__init__()
        self.border = border
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")

        if img1.dtype != img2.dtype:
            img2 = img2.type(img1.dtype)
        if self.border > 0:
            img1 = img1[..., self.border : -self.border, self.border : -self.border]
            img2 = img2[..., self.border : -self.border, self.border : -self.border]

        if img1.size(1) == 3:  # RGB image
            ssims = []
            for i in range(3):
                ssim = self._ssim(img1[:, i : i + 1, :, :], img2[:, i : i + 1, :, :])
                ssims.append(ssim)
            ssim = torch.stack(ssims, dim=1).mean(1)
        else:  # Grayscale
            ssim = self._ssim(img1, img2)

        if self.size_average:
            return ssim.mean()
        else:
            return ssim

    def _ssim(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = (
                self.create_window(self.window_size, channel)
                .to(img1.device)
                .type(img1.dtype)
            )
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean([1, 2, 3])


@dataclass
class DegradationOutput:
    H_img: torch.Tensor
    L_img: torch.Tensor
    R_img: torch.Tensor
    mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    k: torch.Tensor = field(default_factory=lambda: torch.tensor([[[1.0]]]))
    sigma: torch.Tensor = field(default_factory=lambda: torch.tensor([[[0.0]]]))
    sr: float = 1.0
    sf: int = 1
    type: int = 1

    def __post_init__(self):
        if self.mask.numel() == 0:
            self.mask = torch.ones_like(self.R_img)
        device = self.R_img.device
        self.k = self.k.to(device)
        self.sigma = self.sigma.to(device)


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
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        if title != "":
            ax.set_title(title)
        ax.axis("off")
    for ax in axes[num_images:]:
        ax.remove()
    plt.tight_layout()
    plt.show()
