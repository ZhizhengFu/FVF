import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from numpy.typing import NDArray
from .utils import DegradationType
from .utils_image import uint2tensor


class KernelSynthesizer:
    def __init__(
        self,
        trajectory_length: int = 250,
        base_kernel_size: int = 25,
    ):
        self.T = trajectory_length
        self.base_size = base_kernel_size

    def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
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

    def _rotation_matrix(self, angles: torch.Tensor) -> torch.Tensor:
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
            sigma = 0.1 + 6 * torch.rand(1).item()
            kernel = self.gaussian_kernel(target_size, sigma)

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


def p2o(psf: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """
    Converts a Point Spread Function (PSF) to its corresponding Optical Transfer Function (OTF) in the frequency domain.

    Args:
        psf (Tensor): The Point Spread Function, typically a 2D or higher-dimensional tensor.
        shape (tuple[int, int]): The desired shape of the output OTF. This should be a tuple of two integers representing the height and width.

    Returns:
        Tensor: The Optical Transfer Function (OTF) in the frequency domain, computed using the Fast Fourier Transform (FFT).
    """
    otf = torch.zeros(*psf.shape[:-2], *shape)
    otf[..., : psf.shape[-2], : psf.shape[-1]] = psf
    shifts = [-(size // 2) for size in psf.shape[-2:]]
    otf = torch.roll(otf, shifts=shifts, dims=(-2, -1))
    return torch.fft.fft2(otf)


def upsample(x: torch.Tensor, sf: int = 3) -> torch.Tensor:
    """s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    """
    shape = list(x.shape)
    new_shape = shape[:-2] + [shape[-2] * sf, shape[-1] * sf]
    z = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
    z[..., ::sf, ::sf].copy_(x)
    return z


def sisr_pipeline(
    H_img: NDArray[np.uint8],
    sf: int,
    k_synthesizer: KernelSynthesizer,
    device: torch.device,
    k_size: int = 25,
    sigma_max: float = 25,
):
    H_img_tensor = uint2tensor(H_img).to(device)
    sigma = (
        torch.tensor(0.0)
        if torch.randint(0, 9, (1,)).item() == 1
        else torch.empty(1).uniform_(0, sigma_max / 255.0)
    ).to(device)
    k = (
        (
            k_synthesizer.gen_gaussian_kernel(target_size=k_size)
            if torch.randint(0, 8, (1,)).item() > 3
            else k_synthesizer.gen_motion_kernel(target_size=k_size)
        )
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(H_img_tensor.shape[0], -1, -1, -1)
    )
    pad_size = k.shape[0] // 2
    L_img_tensor = F.pad(
        H_img_tensor, (pad_size, pad_size, pad_size, pad_size), mode="circular"
    ).unsqueeze(0)
    L_img_tensor = torch.conv2d(L_img_tensor, k, groups=L_img_tensor.shape[1])[
        ..., 0::sf, 0::sf
    ].squeeze()
    L_img_tensor = L_img_tensor + sigma * torch.randn_like(L_img_tensor)
    w, h = L_img_tensor.shape[-2:]
    FB = p2o(k, (w * sf, h * sf))
    F2B = torch.pow(torch.abs(FB), 2)
    FBC = torch.conj(FB)
    STy = upsample(L_img_tensor, sf)
    FBFy = FBC * torch.fft.fft2(STy)
    R_img_tensor = torch.nn.functional.interpolate(
        L_img_tensor.unsqueeze(0), scale_factor=sf, mode="nearest"
    ).squeeze()
    H_img_tensor = H_img_tensor[..., : R_img_tensor.shape[-2], : R_img_tensor.shape[-1]]
    return (
        DegradationType.SISR,
        H_img_tensor,
        L_img_tensor,
        R_img_tensor,
        k,
        sigma.view([1, 1, 1]),
        sf,
        FB,
        FBC,
        F2B,
        FBFy,
    )


def main():
    from .utils_image import imshow, tensor2float, imread_uint_3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = imread_uint_3(Path("src/utils/test.png"))
    synthesizer = KernelSynthesizer()
    sf = 3
    type, H_img, L_img, R_img, k, sigma, sf, FB, FBC, F2B, FBFy = sisr_pipeline(
        image, sf, synthesizer, device
    )
    imshow(
        [
            tensor2float(H_img),
            tensor2float(L_img),
            tensor2float(R_img),
            tensor2float(k[0].squeeze().unsqueeze(0)),
        ]
    )
    print(H_img.shape, L_img.shape, R_img.shape, k.shape)
    print(sigma.squeeze())


if __name__ == "__main__":
    main()
