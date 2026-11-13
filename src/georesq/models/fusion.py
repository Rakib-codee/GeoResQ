"""Modular Dual-Stream Optical-SAR Segmentation & Controlled Ablation Architecture.

Supports four controlled configurations sharing the exact same ResNet encoder
blocks and multi-scale segmentation decoder:
  1. `optical_only`       : Pre-event optical stream only (no post-SAR input)
  2. `sar_only`           : Post-event SAR stream only (no pre-optical input)
  3. `concat`             : Dual-stream ResNet + naive multi-scale [O_i, S_i] 1x1 concat
  4. `gated_cross_fusion` : Dual-stream ResNet + Pre-Event Optical Prior-Gated SAR &
                            Cross-Modal Difference Fusion with SE recalibration
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_RESNET_URLS: dict[str, str] = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
}

RESNET_LAYERS: dict[str, list[int]] = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
}

VALID_FUSION_MODES = ("optical_only", "sar_only", "concat", "gated_cross_fusion")


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock matching torchvision state_dict keys."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNetBackbone(nn.Module):
    """5-stage ResNet feature pyramid extractor (strides 2, 4, 8, 16, 32)."""

    out_channels: tuple[int, int, int, int, int] = (64, 64, 128, 256, 512)

    def __init__(
        self,
        backbone_name: str = "resnet18",
        in_channels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if backbone_name not in RESNET_LAYERS:
            raise ValueError(
                f"Unsupported backbone_name={backbone_name!r}; choose from {list(RESNET_LAYERS)}"
            )
        layers = RESNET_LAYERS[backbone_name]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self._init_weights()
        if pretrained:
            self._load_imagenet_weights(backbone_name, in_channels)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        block_list = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            block_list.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*block_list)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _load_imagenet_weights(self, backbone_name: str, in_channels: int) -> None:
        state_dict = torch.hub.load_state_dict_from_url(
            IMAGENET_RESNET_URLS[backbone_name], progress=False, map_location="cpu"
        )
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("fc.")
        }
        if in_channels == 1 and "conv1.weight" in state_dict:
            state_dict["conv1.weight"] = state_dict["conv1.weight"].mean(
                dim=1, keepdim=True
            )
        elif in_channels != 3 and "conv1.weight" in state_dict:
            del state_dict["conv1.weight"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if unexpected:
            raise RuntimeError(f"Unexpected ResNet keys: {unexpected}")
        if in_channels in (1, 3) and real_missing:
            raise RuntimeError(f"Missing ResNet keys: {real_missing}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.relu(self.bn1(self.conv1(x)))  # 1/2
        s1 = self.layer1(self.maxpool(s0))       # 1/4
        s2 = self.layer2(s1)                     # 1/8
        s3 = self.layer3(s2)                     # 1/16
        s4 = self.layer4(s3)                     # 1/32
        return [s0, s1, s2, s3, s4]


class ConcatFusionBlock(nn.Module):
    """Naive dual-stream baseline: 1x1 projection over concatenated [O_i, S_i]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_opt: torch.Tensor, feat_sar: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([feat_opt, feat_sar], dim=1))


class SqueezeExcite(nn.Module):
    """Lightweight channel recalibration for fused optical-SAR representations."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class PriorGatedCrossFusionBlock(nn.Module):
    """Dedicated Optical-Prior-Gated SAR & Cross-Modal Difference Fusion Block.

    Physical motivation:
      1. `sar_align`: Projects SAR backscatter features into optical feature space.
      2. `opt_prior_gate`: Spatial gate derived from pre-event optical features to
         suppress background SAR speckle clutter outside building footprints.
      3. `cross_change_gate`: Gates |O_i - S_tilde_i| structural change cues to
         distinguish intact corner reflections from collapsed rubble scatter.
      4. Residual skip from `feat_opt` preserves crisp pre-disaster building edges.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.sar_align = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.opt_prior_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.cross_change_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SqueezeExcite(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat_opt: torch.Tensor, feat_sar: torch.Tensor) -> torch.Tensor:
        sar_aligned = self.sar_align(feat_sar)
        gated_sar = sar_aligned * self.opt_prior_gate(feat_opt)
        diff = torch.abs(feat_opt - sar_aligned)
        gated_diff = diff * self.cross_change_gate(torch.cat([feat_opt, sar_aligned], dim=1))
        merged = self.fuse(torch.cat([feat_opt, gated_sar, gated_diff], dim=1))
        return self.act(feat_opt + self.se(merged))


class DecoderStage(nn.Module):
    """Top-down bilinear upsample + skip fusion + 2x ConvBNReLU refinement."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x_up, skip], dim=1))


class MultiScaleSegmentationDecoder(nn.Module):
    """Pyramid segmentation decoder restoring full (H, W) building damage masks."""

    def __init__(
        self,
        enc_channels: tuple[int, int, int, int, int] = (64, 64, 128, 256, 512),
        num_classes: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = enc_channels
        self.up4 = DecoderStage(c4, c3, 256)  # 1/32 -> 1/16
        self.up3 = DecoderStage(256, c2, 128) # 1/16 -> 1/8
        self.up2 = DecoderStage(128, c1, 64)  # 1/8  -> 1/4
        self.up1 = DecoderStage(64, c0, 64)   # 1/4  -> 1/2
        self.final_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        f0, f1, f2, f3, f4 = features
        d4 = self.up4(f4, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)
        d0 = F.interpolate(d1, size=output_size, mode="bilinear", align_corners=False)
        return self.final_head(d0)


class DualStreamDamageNet(nn.Module):
    """Modular Optical-SAR Dual-Encoder Segmentation Network for GeoResQ."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        fusion_mode: str = "gated_cross_fusion",
        num_classes: int = 4,
        opt_in_channels: int = 3,
        sar_in_channels: int = 3,
        pretrained: bool = True,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_mode not in VALID_FUSION_MODES:
            raise ValueError(
                f"Invalid fusion_mode={fusion_mode!r}; expected one of {VALID_FUSION_MODES}"
            )
        self.backbone_name = backbone_name
        self.fusion_mode = fusion_mode
        self.num_classes = num_classes

        self.optical_encoder: ResNetBackbone | None = None
        self.sar_encoder: ResNetBackbone | None = None
        self.fusion_blocks: nn.ModuleList | None = None

        if fusion_mode in ("optical_only", "concat", "gated_cross_fusion"):
            self.optical_encoder = ResNetBackbone(
                backbone_name=backbone_name,
                in_channels=opt_in_channels,
                pretrained=pretrained,
            )

        if fusion_mode in ("sar_only", "concat", "gated_cross_fusion"):
            self.sar_encoder = ResNetBackbone(
                backbone_name=backbone_name,
                in_channels=sar_in_channels,
                pretrained=pretrained,
            )

        enc_channels = ResNetBackbone.out_channels
        if fusion_mode == "concat":
            self.fusion_blocks = nn.ModuleList(
                [ConcatFusionBlock(ch) for ch in enc_channels]
            )
        elif fusion_mode == "gated_cross_fusion":
            self.fusion_blocks = nn.ModuleList(
                [PriorGatedCrossFusionBlock(ch) for ch in enc_channels]
            )

        self.decoder = MultiScaleSegmentationDecoder(
            enc_channels=enc_channels,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        pre_opt: torch.Tensor,
        post_sar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if post_sar is None:
            if pre_opt.shape[1] != 6:
                raise ValueError(
                    "DualStreamDamageNet expects (pre_opt, post_sar) or 6-ch tensor."
                )
            pre_opt, post_sar = pre_opt[:, :3], pre_opt[:, 3:]

        out_hw = (pre_opt.shape[-2], pre_opt.shape[-1])

        if self.fusion_mode == "optical_only":
            assert self.optical_encoder is not None
            fused_pyramid = self.optical_encoder(pre_opt)
        elif self.fusion_mode == "sar_only":
            assert self.sar_encoder is not None
            fused_pyramid = self.sar_encoder(post_sar)
        else:
            assert self.optical_encoder is not None
            assert self.sar_encoder is not None
            assert self.fusion_blocks is not None
            opt_feats = self.optical_encoder(pre_opt)
            sar_feats = self.sar_encoder(post_sar)
            fused_pyramid = [
                block(o_f, s_f)
                for block, o_f, s_f in zip(self.fusion_blocks, opt_feats, sar_feats)
            ]

        return self.decoder(fused_pyramid, output_size=out_hw)
