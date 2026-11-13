"""GeoResQ segmentation architectures & unified model factory."""

from __future__ import annotations

from georesq.models.fusion import VALID_FUSION_MODES, DualStreamDamageNet
from georesq.models.unet import BaselineUNet


def build_model(model_cfg: dict[str, object]):
    """Instantiate any GeoResQ baseline or dual-stream fusion architecture from config."""
    arch = str(model_cfg.get("architecture", "baseline_unet")).lower()
    num_classes = int(model_cfg.get("num_classes", 4))  # type: ignore[arg-type]

    if arch == "baseline_unet":
        return BaselineUNet(
            in_channels=int(model_cfg.get("in_channels", 6)),  # type: ignore[arg-type]
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 64)),  # type: ignore[arg-type]
        )

    if arch == "dual_stream_unet":
        sar_ch = 3 if bool(model_cfg.get("stack_sar_3ch", True)) else 1
        return DualStreamDamageNet(
            backbone_name=str(model_cfg.get("backbone", "resnet18")),
            fusion_mode=str(model_cfg.get("fusion_mode", "gated_cross_fusion")),
            num_classes=num_classes,
            opt_in_channels=3,
            sar_in_channels=sar_ch,
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout_p=float(model_cfg.get("dropout_p", 0.1)),  # type: ignore[arg-type]
        )

    raise ValueError(f"Unknown model architecture: {arch!r}")


def get_parameter_groups(
    model,
    base_lr: float,
    backbone_lr_scale: float = 0.33,
    weight_decay: float = 1e-3,
) -> list[dict[str, object]]:
    """Use gentler LR on pretrained encoders and full LR on fusion/decoder heads."""
    encoder_params = []
    head_params = []
    encoder_ids: set[int] = set()

    for attr_name in ("optical_encoder", "sar_encoder"):
        enc = getattr(model, attr_name, None)
        if enc is not None:
            for p in enc.parameters():
                if p.requires_grad:
                    encoder_params.append(p)
                    encoder_ids.add(id(p))

    for p in model.parameters():
        if p.requires_grad and id(p) not in encoder_ids:
            head_params.append(p)

    if not encoder_params:
        return [{"params": head_params, "lr": base_lr, "weight_decay": weight_decay}]

    return [
        {
            "params": encoder_params,
            "lr": base_lr * backbone_lr_scale,
            "weight_decay": weight_decay,
        },
        {
            "params": head_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
    ]


__all__ = [
    "BaselineUNet",
    "DualStreamDamageNet",
    "VALID_FUSION_MODES",
    "build_model",
    "get_parameter_groups",
]
