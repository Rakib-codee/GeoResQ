"""Official BRIGHT Benchmark-Compatible Baseline U-Net.

Matches `bda_benchmark/model/UNet.py` (Ronneberger et al., MICCAI 2015) with
5 encoder stages [64, 128, 256, 512, 1024] and transposed-convolution decoder.
Accepts `(pre_opt, post_sar)` tensors and concatenates along channel dim -> 6 channels.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


if nn is not None:

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class BaselineUNet(nn.Module):
        """Early-concatenation 6-channel U-Net matching official BRIGHT baseline."""

        def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = 4,
            base_channels: int = 64,
        ) -> None:
            super().__init__()
            c1 = base_channels
            c2 = c1 * 2
            c3 = c2 * 2
            c4 = c3 * 2
            c5 = c4 * 2

            self.encoder1 = ConvBlock(in_channels, c1)
            self.encoder2 = ConvBlock(c1, c2)
            self.encoder3 = ConvBlock(c2, c3)
            self.encoder4 = ConvBlock(c3, c4)
            self.encoder5 = ConvBlock(c4, c5)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
            self.decoder4 = ConvBlock(c4 * 2, c4)

            self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
            self.decoder3 = ConvBlock(c3 * 2, c3)

            self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
            self.decoder2 = ConvBlock(c2 * 2, c2)

            self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
            self.decoder1 = ConvBlock(c1 * 2, c1)

            self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

        def forward(
            self,
            pre_opt: torch.Tensor,
            post_sar: torch.Tensor | None = None,
        ) -> torch.Tensor:
            x = (
                torch.cat([pre_opt, post_sar], dim=1)
                if post_sar is not None
                else pre_opt
            )
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool(enc1))
            enc3 = self.encoder3(self.pool(enc2))
            enc4 = self.encoder4(self.pool(enc3))
            enc5 = self.encoder5(self.pool(enc4))

            dec4 = self.decoder4(torch.cat([self.upconv4(enc5), enc4], dim=1))
            dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
            dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
            dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
            return self.final_conv(dec1)
