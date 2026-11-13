"""Official BRIGHT Benchmark Loss (CE + 0.75 * Lovasz-Softmax) & Auxiliary Localization Loss.

References:
- Berman et al., "The Lovasz-Softmax loss", CVPR 2018 (MIT License).
- Official BRIGHT `bda_benchmark/script/standard_ML/train_UNet.py`.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t. sorted pixel errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes: str = "present"):
    """Multi-class Lovasz-Softmax on flattened valid pixels [P, C]."""
    if probas.numel() == 0:
        return probas.sum() * 0.0

    num_classes = probas.size(1)
    losses = []
    class_indices = (
        list(range(num_classes)) if classes in ("all", "present") else list(classes)
    )
    for c in class_indices:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    probas,
    labels,
    classes: str = "present",
    ignore_index: int = 255,
):
    """Multi-class Lovasz-Softmax loss for [B, C, H, W] probabilities."""
    b, c, h, w = probas.size()
    flat_probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    flat_labels = labels.view(-1)
    valid = flat_labels != ignore_index
    return lovasz_softmax_flat(flat_probas[valid], flat_labels[valid], classes=classes)


if nn is not None:

    class BrightComboLoss(nn.Module):
        """CE + lovasz_weight * Lovasz-Softmax (default lovasz_weight=0.75 matches BRIGHT)."""

        def __init__(
            self,
            class_weights: list[float] | None = None,
            lovasz_weight: float = 0.75,
            ignore_index: int = 255,
        ) -> None:
            super().__init__()
            self.lovasz_weight = float(lovasz_weight)
            self.ignore_index = int(ignore_index)
            if class_weights is not None:
                self.register_buffer(
                    "weight_tensor", torch.tensor(class_weights, dtype=torch.float32)
                )
            else:
                self.weight_tensor = None

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(
                logits,
                targets,
                weight=self.weight_tensor,
                ignore_index=self.ignore_index,
            )
            probas = F.softmax(logits, dim=1)
            lov = lovasz_softmax(probas, targets, ignore_index=self.ignore_index)
            return ce + self.lovasz_weight * lov

    class BuildingLocBCEDiceLoss(nn.Module):
        """Binary BCEWithLogits + Soft Dice on building footprint mask (ignoring 255)."""

        def __init__(self, ignore_index: int = 255, smooth: float = 1.0) -> None:
            super().__init__()
            self.ignore_index = int(ignore_index)
            self.smooth = float(smooth)

        def forward(
            self, loc_logits: torch.Tensor, loc_targets: torch.Tensor
        ) -> torch.Tensor:
            flat_logits = loc_logits.view(-1)
            flat_targets = loc_targets.view(-1)
            valid = flat_targets != self.ignore_index
            if not torch.any(valid):
                return loc_logits.sum() * 0.0

            v_logits = flat_logits[valid]
            v_targets = (flat_targets[valid] > 0).float()

            bce = F.binary_cross_entropy_with_logits(v_logits, v_targets)
            probs = torch.sigmoid(v_logits)
            inter = (probs * v_targets).sum()
            dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
                probs.sum() + v_targets.sum() + self.smooth
            )
            return bce + dice_loss
