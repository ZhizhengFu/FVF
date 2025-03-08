import torch
from torch import Tensor


# modify from https://github.com/cszn/KAIR/blob/master/utils/utils_sisr.py#L425
def p2o(psf: Tensor, shape: tuple[int, int]) -> Tensor:
    """
    Converts a Point Spread Function (PSF) to its corresponding Optical Transfer Function (OTF) in the frequency domain.

    Args:
        psf (Tensor): The Point Spread Function, typically a 2D or higher-dimensional tensor.
        shape (tuple[int, int]): The desired shape of the output OTF. This should be a tuple of two integers representing the height and width.

    Returns:
        Tensor: The Optical Transfer Function (OTF) in the frequency domain, computed using the Fast Fourier Transform (FFT).
    """
    otf = torch.zeros(*psf.shape[:-2], *shape, dtype=psf.dtype, device=psf.device)
    otf[..., : psf.shape[-2], : psf.shape[-1]] = psf
    shifts = [-(size // 2) for size in psf.shape[-2:]]
    otf = torch.roll(otf, shifts=shifts, dims=(-2, -1))
    return torch.fft.fftn(otf, dim=(-2, -1))


def upsample(x, sf=3):
    """s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(
        x
    )
    z[..., st::sf, st::sf].copy_(x)
    return z


def pre_calculate(x, k, sf):
    """
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    """
    w, h = x.shape[-2:]
    FB = p2o(k, (w * sf, h * sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def main():
    psf = torch.rand(1, 1, 256, 256)
    shape = (512, 512)
    otf_torch = p2o(psf, shape)
    print(otf_torch)


if __name__ == "__main__":
    main()
