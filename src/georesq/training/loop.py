"""Reusable Reproducible Training, Checkpoint-Resume, Config, and Evaluation Engine."""

from __future__ import annotations

import ast
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from georesq.evaluation.metrics import DamageEvaluator


def _parse_scalar(val_str: str) -> object:
    low = val_str.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    if (val_str.startswith('"') and val_str.endswith('"')) or (
        val_str.startswith("'") and val_str.endswith("'")
    ):
        return val_str[1:-1]
    if val_str.startswith("[") and val_str.endswith("]"):
        return ast.literal_eval(val_str)
    try:
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            return val_str


def load_yaml_config(path: Path) -> dict[str, object]:
    """Load YAML config via PyYAML if installed, else zero-dependency fallback."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        root: dict[str, object] = {}
        stack: list[tuple[int, object]] = [(-1, root)]
        lines = text.splitlines()
        for idx, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if stripped.startswith("- "):
                assert isinstance(parent, list)
                parent.append(_parse_scalar(stripped[2:].strip()))
            else:
                k_part, _, v_part = stripped.partition(":")
                key: object = int(k_part) if k_part.isdigit() else k_part.strip()
                v_str = v_part.strip()
                if v_str and not v_str.startswith(('"', "'")) and " #" in v_str:
                    v_str = v_str.split(" #", 1)[0].strip()
                if v_str == "":
                    nxt: object = {}
                    for j in range(idx + 1, len(lines)):
                        ns = lines[j].strip()
                        if not ns or ns.startswith("#"):
                            continue
                        if ns.startswith("- "):
                            nxt = []
                        break
                    assert isinstance(parent, dict)
                    parent[key] = nxt  # type: ignore[index]
                    stack.append((indent, nxt))
                else:
                    assert isinstance(parent, dict)
                    parent[key] = _parse_scalar(v_str)  # type: ignore[index]
        return root


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def count_model_parameters(model) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total * 4) / (1024 ** 2)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "fp32_size_mb": round(float(size_mb), 2),
    }


def evaluate_model(model, loader, criterion, device, use_amp: bool = False) -> dict[str, object]:
    import torch

    model.eval()
    evaluator = DamageEvaluator(num_classes=4, ignore_index=255)
    total_loss = 0.0
    num_batches = 0
    num_images = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            pre = batch["pre_opt"].to(device, non_blocking=True)
            post = batch["post_sar"].to(device, non_blocking=True)
            targets = batch["clf_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(pre, post)
            loss = criterion(logits.float(), targets)
            preds = torch.argmax(logits, dim=1)

            total_loss += float(loss.item())
            num_batches += 1
            num_images += int(pre.size(0))
            evaluator.update(targets.cpu().numpy(), preds.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(1e-6, time.perf_counter() - t0)

    summary = evaluator.compute_summary()
    summary["loss"] = round(total_loss / max(1, num_batches), 6)
    summary["inference_seconds"] = round(elapsed, 4)
    summary["ms_per_tile"] = round(1000.0 * elapsed / max(1, num_images), 2)
    summary["tiles_per_sec"] = round(num_images / elapsed, 2)
    return summary


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score: float,
    history: list[dict[str, object]],
    extra_meta: dict[str, object] | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_score": float(best_score),
        "history": history,
        "meta": extra_meta or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device=None,
) -> dict[str, object]:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def save_training_curves(history: list[dict[str, object]], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(h["epoch"]) for h in history]
    tr_loss = [float(h["train_loss"]) for h in history]
    va_loss = [float(h["val_loss"]) for h in history]
    va_miou4 = [float(h["val_mIoU_all4"]) * 100.0 for h in history]
    va_miou3 = [float(h["val_mIoU_foreground3"]) * 100.0 for h in history]
    va_f1 = [float(h["val_macro_f1_foreground3"]) * 100.0 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(epochs, tr_loss, marker="o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, va_loss, marker="s", markersize=3, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combo Loss (CE + 0.75 Lovasz)")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, va_miou4, marker="o", markersize=3, label="Val mIoU (All 4)%")
    axes[1].plot(epochs, va_miou3, marker="s", markersize=3, label="Val mIoU (3 Damage Classes)%")
    axes[1].plot(epochs, va_f1, marker="^", markersize=3, label="Val Macro-F1 (3 Damage Classes)%")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Validation Segmentation Quality")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_history_csv(history: list[dict[str, object]], csv_path: Path) -> None:
    if not history:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
