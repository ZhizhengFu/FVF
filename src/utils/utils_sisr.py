import torch
from torch import Tensor

# modify from https://github.com/cszn/KAIR/blob/master/utils/utils_sisr.py#L425
def p2o(psf: Tensor, shape: tuple[int, int]) -> Tensor:
    otf = torch.zeros(*psf.shape[:-2], *shape, dtype=psf.dtype, device=psf.device)
    otf[..., :psf.shape[-2], :psf.shape[-1]] = psf
    shifts = [-(size // 2) for size in psf.shape[-2:]]
    otf = torch.roll(otf, shifts=shifts, dims=(-2, -1))
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf

def main():

    psf = torch.rand(1, 1, 256, 256)
    shape = (512, 512)

    otf_torch = p2o(psf, shape)
    print(otf_torch)

if __name__ == "__main__":
    main()
