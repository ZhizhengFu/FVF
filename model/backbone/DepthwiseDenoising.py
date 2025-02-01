import torch
import torch.nn as nn

class AdaptiveModulation(nn.Module):
    def __init__(self, channels):
        super(AdaptiveModulation, self).__init__()
        self.gamma_fc = nn.Linear(1, channels)
        self.beta_fc = nn.Linear(1, channels)

    def forward(self, x, beta):
        beta = beta.view(-1, 1)  # (N, 1)
        gamma = self.gamma_fc(beta).view(-1, x.shape[1], 1, 1)  # (N, C, 1, 1)
        beta = self.beta_fc(beta).view(-1, x.shape[1], 1, 1)   # (N, C, 1, 1)
        return gamma * x + beta

# 示例
modulation = AdaptiveModulation(64)
x = torch.randn(1, 64, 128, 128)
beta = torch.tensor([sf / (sigma + 1)])  # 计算 beta
y = modulation(x, beta)
print(y.shape)


class DepthwiseDenoisingBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, scale=0.1):
        super(DepthwiseDenoisingBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size//2, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.activation = nn.SiLU()
        self.scale = scale

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return residual + self.scale * x

class DenoisingNetWithUpsampleDownsample(nn.Module):
    def __init__(self, channels=64, scale=0.1):
        super(DenoisingNetWithUpsampleDownsample, self).__init__()

        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.block1 = DepthwiseDenoisingBlock(channels, scale=scale)
        self.block2 = DepthwiseDenoisingBlock(channels, scale=scale)
        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.upsample(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128)
    denoise_block = DenoisingNetWithUpsampleDownsample(64, scale=0.1)
    y = denoise_block(x)

    print(y.shape)  # (1, 64, 128, 128)
