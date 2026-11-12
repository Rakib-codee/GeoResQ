#!/usr/bin/env python3
"""GeoResQ Environment, GPU, Storage & Dataset-Access Audit Script.

Works with zero third-party dependencies on local macOS (for lightweight validation)
and detects full GPU/VRAM/package/network details on Kaggle Notebooks or Google Colab.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CANDIDATE_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "imageio",
    "tifffile",
    "rasterio",
    "PyYAML",
    "matplotlib",
    "tqdm",
    "scikit-learn",
    "albumentations",
]

HF_TARGET_ZIP_URL = (
    "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main/target.zip"
)


def detect_environment() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if "COLAB_GPU" in os.environ or Path("/content").exists():
        return "colab"
    if sys.platform == "darwin":
        return "local_macos"
    return "linux_other"


def get_system_ram_gb() -> float | None:
    try:
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 2)
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out.strip()) / (1024 ** 3), 2)
    except Exception:
        return None
    return None


def get_disk_info(target_dir: Path) -> dict[str, float | str]:
    check_dir = target_dir if target_dir.exists() else Path.cwd()
    usage = shutil.disk_usage(check_dir)
    return {
        "checked_path": str(check_dir.resolve()),
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
    }


def get_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in CANDIDATE_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def get_gpu_info() -> dict[str, object]:
    try:
        import torch  # type: ignore
    except ImportError:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
        }

    cuda_ok = bool(torch.cuda.is_available())
    devices = []
    if cuda_ok:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024 ** 3), 2),
                }
            )
    return {
        "torch_available": True,
        "cuda_available": cuda_ok,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": len(devices),
        "devices": devices,
    }


def verify_hf_range_support(timeout: int = 20) -> dict[str, object]:
    try:
        req = urllib.request.Request(
            HF_TARGET_ZIP_URL, headers={"Range": "bytes=0-15"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", resp.getcode())
            content_range = resp.headers.get("Content-Range", "")
            data = resp.read()
        return {
            "reachable": True,
            "http_status": status,
            "partial_content_206": status == 206,
            "bytes_read": len(data),
            "content_range": content_range,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "http_status": None,
            "partial_content_206": False,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit GeoResQ execution environment.")
    parser.add_argument(
        "--verify-hf-range",
        action="store_true",
        help="Verify HTTP 206 byte-range access to official BRIGHT Hugging Face archive.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save audit report as JSON.",
    )
    args = parser.parse_args()

    env_name = detect_environment()
    storage_dir = Path("/kaggle/working") if env_name == "kaggle" else Path.cwd()

    report: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env_name,
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "system_ram_gb": get_system_ram_gb(),
        "disk": get_disk_info(storage_dir),
        "gpu": get_gpu_info(),
        "packages": get_package_versions(),
    }

    if args.verify_hf_range:
        report["bright_hf_range_check"] = verify_hf_range_support()

    formatted = json.dumps(report, indent=2)
    print(formatted)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted + "\n", encoding="utf-8")
        print(f"\nSaved environment audit report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
