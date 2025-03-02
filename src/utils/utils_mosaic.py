import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


def dm(imgs):
    """bilinear demosaicking
    Args:
        imgs: Nx4xW/2xH/2
    Returns:
        output: Nx3xWxH
    """
    k_r = 1 / 4 * torch.FloatTensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).type_as(imgs)
    k_g = 1 / 4 * torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]).type_as(imgs)
    k = torch.stack((k_r, k_g, k_r), dim=0).unsqueeze(1)

    rgb = torch.zeros(imgs.size(0), 3, imgs.size(2) * 2, imgs.size(3) * 2).type_as(imgs)
    rgb[:, 0, 0::2, 0::2] = imgs[:, 0, :, :]
    rgb[:, 1, 0::2, 1::2] = imgs[:, 1, :, :]
    rgb[:, 1, 1::2, 0::2] = imgs[:, 2, :, :]
    rgb[:, 2, 1::2, 1::2] = imgs[:, 3, :, :]

    rgb = nn.functional.pad(rgb, (1, 1, 1, 1), mode="circular")
    rgb = nn.functional.conv2d(rgb, k, groups=3, padding=0, bias=None)

    return rgb


def mosaic_CFA_Bayer(RGB):
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2])
    mask = np.concatenate(
        (R_m[..., np.newaxis], G_m[..., np.newaxis], B_m[..., np.newaxis]), axis=-1
    )
    mosaic = np.multiply(mask, RGB)  # mask*RGB
    CFA = mosaic.sum(2).astype(np.uint8)

    CFA4 = np.zeros((RGB.shape[0] // 2, RGB.shape[1] // 2, 4), dtype=np.uint8)
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]

    return CFA, CFA4, mosaic, mask


def masks_CFA_Bayer(shape):
    pattern = "RGGB"
    channels = dict((channel, np.zeros(shape)) for channel in "RGB")
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    return tuple(channels[c].astype(bool) for c in "RGB")


# Load and process the image
Im = cv2.imread("src/utils/test.bmp", cv2.IMREAD_COLOR)
Im = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)

# Get mosaic and CFA
CFA, CFA4, mosaic, mask = mosaic_CFA_Bayer(Im)

# Convert CFA4 to a 4D tensor
CFA4_tensor = torch.from_numpy(CFA4).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# Demosaicking using the function
demosaicked_image = dm(CFA4_tensor)

# Convert the tensor back to numpy for visualization
demosaicked_image_np = (
    demosaicked_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
)

# Display images using matplotlib
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(CFA)
axes[0].set_title("CFA")
axes[1].imshow(mosaic)
axes[1].set_title("Mosaic")
axes[2].imshow(mask.astype(np.float32))
axes[2].set_title("Mask")
axes[3].imshow(demosaicked_image_np)
axes[3].set_title("Demosaicked Image")

# Remove axes for clarity
for ax in axes:
    ax.axis("off")

plt.show()
