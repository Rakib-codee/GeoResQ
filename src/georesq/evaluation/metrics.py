"""Research-grade Segmentation & Damage Assessment Evaluator (Pure NumPy).

Computes both:
1. Full GeoResQ research metrics (Section 10):
   - 4x4 Confusion Matrix
   - mIoU (all 4 classes) & mIoU_foreground (Intact, Damaged, Destroyed)
   - Class-wise IoU, Precision, Recall, F1
   - Macro F1 (all 4 classes) & Macro F1 (3 building damage classes)
2. Official BRIGHT benchmark metrics (`bda_benchmark/script/standard_ML/train_UNet.py`):
   - Binary building localization F1 (`loc_f1`)
   - Harmonic mean damage F1 across [Intact, Damaged, Destroyed] (`harmonic_damage_f1`)
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ("background", "intact", "damaged", "destroyed")


class DamageEvaluator:
    def __init__(self, num_classes: int = 4, ignore_index: int = 255) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.eps = 1e-7
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.loc_confusion = np.zeros((2, 2), dtype=np.int64)

    def update(self, targets: np.ndarray, preds: np.ndarray) -> None:
        """Accumulate batch predictions and ground-truth masks (shape [H,W] or [B,H,W])."""
        gt = np.asarray(targets, dtype=np.int64).ravel()
        pr = np.asarray(preds, dtype=np.int64).ravel()

        valid = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_classes)
        gt_v = gt[valid]
        pr_v = np.clip(pr[valid], 0, self.num_classes - 1)

        encoded = self.num_classes * gt_v + pr_v
        counts = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        gt_loc = (gt_v > 0).astype(np.int64)
        pr_loc = (pr_v > 0).astype(np.int64)
        loc_counts = np.bincount(2 * gt_loc + pr_loc, minlength=4)
        self.loc_confusion += loc_counts.reshape(2, 2)

    def compute_summary(self) -> dict[str, object]:
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + self.eps)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2.0 * precision * recall / (precision + recall + self.eps)

        loc_tp = float(self.loc_confusion[1, 1])
        loc_fp = float(self.loc_confusion[0, 1])
        loc_fn = float(self.loc_confusion[1, 0])
        loc_prec = loc_tp / (loc_tp + loc_fp + self.eps)
        loc_rec = loc_tp / (loc_tp + loc_fn + self.eps)
        loc_f1 = 2.0 * loc_prec * loc_rec / (loc_prec + loc_rec + self.eps)

        fg_f1 = f1[1:]
        harmonic_damage_f1 = float(len(fg_f1) / np.sum(1.0 / (fg_f1 + self.eps)))

        overall_acc = float(tp.sum() / max(1.0, cm.sum()))

        return {
            "overall_pixel_accuracy": round(overall_acc, 6),
            "mIoU_all4": round(float(np.mean(iou)), 6),
            "mIoU_foreground3": round(float(np.mean(iou[1:])), 6),
            "macro_f1_all4": round(float(np.mean(f1)), 6),
            "macro_f1_foreground3": round(float(np.mean(f1[1:])), 6),
            "bright_loc_f1": round(float(loc_f1), 6),
            "bright_harmonic_damage_f1": round(harmonic_damage_f1, 6),
            "per_class": {
                CLASS_NAMES[i]: {
                    "iou": round(float(iou[i]), 6),
                    "precision": round(float(precision[i]), 6),
                    "recall": round(float(recall[i]), 6),
                    "f1": round(float(f1[i]), 6),
                    "support_pixels": int(self.confusion_matrix[i].sum()),
                }
                for i in range(self.num_classes)
            },
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


def decompose_confusion_errors(cm_input: list[list[int]] | np.ndarray) -> dict[str, object]:
    """Decompose a 4x4 confusion matrix into physical localization vs. severity error modes."""
    cm = np.asarray(cm_input, dtype=np.float64)
    if cm.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) confusion matrix, got {cm.shape}")

    eps = 1e-9
    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, eps), 0.0)

    gt_bg = float(cm[0, :].sum())
    gt_bldg = float(cm[1:, :].sum())

    false_bldg_alarm_rate = float(cm[0, 1:].sum() / max(gt_bg, eps))
    missed_bldg_rate = float(cm[1:, 0].sum() / max(gt_bldg, eps))

    # Conditional severity grading accuracy among detected building pixels (gt > 0 AND pred > 0)
    detected_bldg_sub = cm[1:, 1:]
    detected_bldg_total = float(detected_bldg_sub.sum())
    cond_severity_acc = (
        float(np.trace(detected_bldg_sub) / detected_bldg_total)
        if detected_bldg_total > 0
        else 0.0
    )

    def _routing_dict(cls_idx: int) -> dict[str, float]:
        tot = float(cm[cls_idx, :].sum())
        if tot <= 0:
            return {f"pred_{name}_pct": 0.0 for name in CLASS_NAMES}
        return {
            f"pred_{name}_pct": round(100.0 * float(cm[cls_idx, j]) / tot, 3)
            for j, name in enumerate(CLASS_NAMES)
        }

    return {
        "row_normalized_pct": np.round(row_norm * 100.0, 3).tolist(),
        "false_building_alarm_rate_pct": round(100.0 * false_bldg_alarm_rate, 3),
        "missed_building_fn_rate_pct": round(100.0 * missed_bldg_rate, 3),
        "conditional_severity_acc_given_detected_pct": round(100.0 * cond_severity_acc, 3),
        "intact_routing_pct": _routing_dict(1),
        "damaged_routing_pct": _routing_dict(2),
        "destroyed_routing_pct": _routing_dict(3),
    }
