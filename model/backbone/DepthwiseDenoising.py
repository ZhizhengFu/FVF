import torch
import torch.nn as nn

class DepthwiseDenoisingBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(DepthwiseDenoisingBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size//2, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm([channels, 1, 1])
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = self.activation(x)
        return x + residual

class DenoisingNetWithUpsampleDownsample(nn.Module):
    def __init__(self, channels=64):
        super(DenoisingNetWithUpsampleDownsample, self).__init__()

        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.block1 = DepthwiseDenoisingBlock(channels)
        self.block2 = DepthwiseDenoisingBlock(channels)
        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.upsample(x)
        return x

if __name__ == "__main__":

    x = torch.randn(1, 64, 128, 128)
    denoise_block = DenoisingNetWithUpsampleDownsample(64)
    y = denoise_block(x)

    print(y.shape)  # (1, 64, 128, 128)
