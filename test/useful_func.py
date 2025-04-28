import torch
import math
# # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
# elif args.task in ['jpeg_car']:
#     img_gt = cv2.imread(path, 0)
#     result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
#     img_lq = cv2.imdecode(encimg, 0)
#     img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
#     img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
# return imgname, img_lq, img_gt
def gen_sf_mask(sf: int, shape):
    mask = torch.zeros_like(shape)
    mask[..., ::sf, ::sf] = 1
    return mask


def FMFH(n: int, mask) -> torch.Tensor:
    k = torch.arange(n)
    exponent = -2j * math.pi * torch.outer(k, k) / n
    selected_columns = torch.exp(exponent)[:, mask]
    return (selected_columns @ selected_columns.H) / n


def FMFH_2d_efficient(height: int, width: int, mask_2d: torch.Tensor) -> torch.Tensor:
    nonzero_indices = torch.nonzero(mask_2d)
    k = len(nonzero_indices)

    y, x = nonzero_indices[:, 0], nonzero_indices[:, 1]
    ky = torch.arange(height, dtype=torch.float32)
    kx = torch.arange(width, dtype=torch.float32)

    basis_y = torch.exp(-2j * math.pi * torch.outer(ky, y) / height) / math.sqrt(height)
    basis_x = torch.exp(-2j * math.pi * torch.outer(kx, x) / width) / math.sqrt(width)

    full_basis = basis_y.unsqueeze(1) * basis_x.unsqueeze(0)
    full_basis = full_basis.reshape(height * width, k)

    return full_basis @ full_basis.H
