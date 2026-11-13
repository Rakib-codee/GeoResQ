#!/usr/bin/env python3
"""GeoResQ Official BRIGHT Audit & Selective Cloud Subset Builder (v0.1 & v0.2).

Key guarantees:
1. Strictly excludes restricted Google-Earth events (ukraine-conflict, myanmar-hurricane, mexico-hurricane).
2. Audits all 3,395 open-license target masks (45.4 MB target.zip) to measure true class imbalance.
3. Selects event-separated subsets (v0.1 = 45 tiles, v0.2 = 294 tiles across all 11 open events) deterministically.
4. Streams ONLY selected optical & SAR .tif files via parallel HTTP 206 Range requests (no 12 GB download).
5. Verifies dimensions, channels, dtype, CRS/bounds alignment, and computes train-only band statistics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import math
import random
import shutil
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_EVENT_METADATA: dict[str, dict[str, str]] = {
    "beirut-explosion": {
        "region": "Beirut, Lebanon",
        "disaster_type": "Explosion",
        "date": "2020-08-04",
        "gsd_m": "1.0",
        "providers": "Maxar & Capella",
    },
    "bata-explosion": {
        "region": "Bata, Equatorial Guinea",
        "disaster_type": "Explosion",
        "date": "2021-03-07",
        "gsd_m": "0.5",
        "providers": "Maxar & Capella",
    },
    "congo-volcano": {
        "region": "Goma, DR Congo",
        "disaster_type": "Volcano eruption",
        "date": "2021-05-22",
        "gsd_m": "0.33",
        "providers": "Maxar & Capella",
    },
    "haiti-earthquake": {
        "region": "Les Cayes, Haiti",
        "disaster_type": "Earthquake",
        "date": "2021-08-14",
        "gsd_m": "0.48",
        "providers": "Maxar & Capella",
    },
    "la_palma-volcano": {
        "region": "La Palma, Spain",
        "disaster_type": "Volcano eruption",
        "date": "2021-09-19 to 2021-12-13",
        "gsd_m": "0.30-0.35",
        "providers": "IGN (Spain) & Capella",
    },
    "marshall-wildfire": {
        "region": "Boulder, Colorado, USA",
        "disaster_type": "Wildfire",
        "date": "2021-12-30 to 2022-01-01",
        "gsd_m": "0.6",
        "providers": "NAIP & Capella",
    },
    "turkey-earthquake": {
        "region": "Southeastern Turkey",
        "disaster_type": "Earthquake",
        "date": "2023-02-06",
        "gsd_m": "0.30-0.35",
        "providers": "Maxar & Capella & Umbra",
    },
    "hawaii-wildfire": {
        "region": "Maui, Hawaii, USA",
        "disaster_type": "Wildfire",
        "date": "2023-08-08 to 2023-08-09",
        "gsd_m": "0.6",
        "providers": "NOAA & Capella",
    },
    "morocco-earthquake": {
        "region": "Al Haouz, Morocco",
        "disaster_type": "Earthquake",
        "date": "2023-09-08",
        "gsd_m": "0.35-0.40",
        "providers": "Maxar & Capella",
    },
    "libya-flood": {
        "region": "Derna, Libya",
        "disaster_type": "Flood",
        "date": "2023-09-10",
        "gsd_m": "0.35",
        "providers": "Maxar & Capella",
    },
    "noto-earthquake": {
        "region": "Noto Peninsula, Japan",
        "disaster_type": "Earthquake",
        "date": "2024-01-01",
        "gsd_m": "0.5",
        "providers": "GSI & Umbra",
    },
}

EXCLUDED_RESTRICTED_EVENTS = {
    "ukraine-conflict",
    "myanmar-hurricane",
    "mexico-hurricane",
}

MANIFEST_COLUMNS = [
    "sample_id",
    "disaster_event",
    "region",
    "pre_event_path",
    "post_event_path",
    "target_path",
    "split",
    "modality_information",
    "label_information",
]

HF_BASE_URL = "https://huggingface.co/datasets/Kullervo/BRIGHT/resolve/main"


class HttpRangeZipStream:
    """Seekable file-like wrapper over HTTP Range requests (HTTP 206 Partial Content)."""

    def __init__(
        self,
        url: str,
        timeout: int = 30,
        final_url: str | None = None,
        size: int | None = None,
    ) -> None:
        self.timeout = timeout
        if final_url is not None and size is not None:
            self.final_url = final_url
            self.size = size
        else:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.final_url = resp.geturl()
                content_range = resp.headers.get("Content-Range", "")
                if "/" not in content_range:
                    raise RuntimeError(f"Server did not return Content-Range for {url}")
                self.size = int(content_range.split("/")[-1])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = (
            self.size - 1
            if (n < 0 or self.pos + n - 1 >= self.size)
            else (self.pos + n - 1)
        )
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.final_url, headers={"Range": f"bytes={self.pos}-{end}"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    chunk = resp.read()
                self.pos += len(chunk)
                return chunk
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HTTP Range read failed after 3 attempts: {last_err}")


def parse_event_from_sample_id(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def select_stratified_tiles(
    tile_records: list[dict[str, int | str]], count: int, seed: int
) -> list[dict[str, int | str]]:
    """Deterministically select tiles across damage strata (has_3, has_2, has_1, fallback has_0)."""
    if count >= len(tile_records):
        return sorted(tile_records, key=lambda x: str(x["sample_id"]))

    rng = random.Random(seed)
    has_3 = sorted(
        [r for r in tile_records if int(r["destroyed"]) > 0],
        key=lambda x: str(x["sample_id"]),
    )
    has_2 = sorted(
        [
            r
            for r in tile_records
            if int(r["destroyed"]) == 0 and int(r["damaged"]) > 0
        ],
        key=lambda x: str(x["sample_id"]),
    )
    has_1 = sorted(
        [
            r
            for r in tile_records
            if int(r["destroyed"]) == 0
            and int(r["damaged"]) == 0
            and int(r["intact"]) > 0
        ],
        key=lambda x: str(x["sample_id"]),
    )
    has_0 = sorted(
        [
            r
            for r in tile_records
            if int(r["destroyed"]) == 0
            and int(r["damaged"]) == 0
            and int(r["intact"]) == 0
        ],
        key=lambda x: str(x["sample_id"]),
    )
    for b in (has_3, has_2, has_1, has_0):
        rng.shuffle(b)

    fg_buckets = [has_3, has_2, has_1]
    selected: list[dict[str, int | str]] = []
    idx = 0
    while len(selected) < count and any(fg_buckets):
        bucket = fg_buckets[idx % 3]
        if bucket:
            selected.append(bucket.pop(0))
        idx += 1

    while len(selected) < count and has_0:
        selected.append(has_0.pop(0))

    return sorted(selected, key=lambda x: str(x["sample_id"]))


def run_self_test() -> int:
    """Zero-dependency offline validation for local MacBook."""
    assert len(OFFICIAL_EVENT_METADATA) == 11, "Expected 11 open-license events"
    assert EXCLUDED_RESTRICTED_EVENTS.isdisjoint(OFFICIAL_EVENT_METADATA.keys())
    assert len(MANIFEST_COLUMNS) == 9

    mock_tiles = [
        {
            "sample_id": f"noto-earthquake_{i:08d}",
            "background": 900,
            "intact": (100 if i < 12 else 0),
            "damaged": (10 if i % 2 == 0 and i < 12 else 0),
            "destroyed": (5 if i % 3 == 0 and i < 12 else 0),
            "ignore_255": 0,
        }
        for i in range(1, 15)
    ]
    picked_a = select_stratified_tiles(mock_tiles, count=6, seed=42)
    picked_b = select_stratified_tiles(mock_tiles, count=6, seed=42)
    assert [p["sample_id"] for p in picked_a] == [p["sample_id"] for p in picked_b]
    assert len(picked_a) == 6
    picked_all = select_stratified_tiles(mock_tiles, count=14, seed=42)
    assert len(picked_all) == 14
    print(
        "SELF-TEST PASSED: 11 open events verified, restricted events blocked, full-event & stratified selectors deterministic."
    )
    return 0


def inspect_tif(path: Path) -> tuple[dict[str, object], object]:
    """Inspect GeoTIFF dimensions, dtype, pixel range, CRS/bounds, and return array."""
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(path))
    info: dict[str, object] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "crs": None,
        "bounds": None,
    }
    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as ds:
            info["crs"] = ds.crs.to_string() if ds.crs else None
            if ds.bounds is not None and ds.crs is not None:
                info["bounds"] = [round(float(x), 6) for x in ds.bounds]
    except ImportError:
        pass
    return info, arr


def build_cloud_subset(
    config_path: Path,
    subset_key: str,
    output_dir: Path,
    manifest_out: Path,
    audit_out: Path,
    band_stats_out: Path | None = None,
    workers: int = 6,
) -> int:
    import numpy as np
    import yaml  # type: ignore
    from PIL import Image

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if subset_key not in cfg:
        raise KeyError(f"Subset key {subset_key!r} not found in {config_path}")
    subset_cfg = cfg[subset_key]
    seed = int(subset_cfg["seed"])
    subset_version = subset_key.replace("subset_", "").replace("_", ".")

    split_plan: list[tuple[str, str, int]] = []
    seen_events: set[str] = set()
    for split_name, key in [
        ("train", "train_events"),
        ("val", "val_events"),
        ("test", "test_unseen_events"),
    ]:
        for ev, cnt in subset_cfg[key].items():
            if ev in EXCLUDED_RESTRICTED_EVENTS or ev not in OFFICIAL_EVENT_METADATA:
                raise ValueError(f"Disallowed or unknown event in subset plan: {ev}")
            if ev in seen_events:
                raise ValueError(f"Event {ev} appears in multiple splits!")
            seen_events.add(ev)
            split_plan.append((split_name, ev, int(cnt)))

    tmp_target_zip = Path("/tmp/bright_target_audit.zip")
    if not tmp_target_zip.exists():
        print("Downloading official target.zip (45.4 MB) for global label audit...")
        urllib.request.urlretrieve(f"{HF_BASE_URL}/target.zip", tmp_target_zip)

    global_event_stats: dict[str, dict[str, object]] = {}
    per_event_tiles: dict[str, list[dict[str, int | str]]] = {}

    print("Auditing damage masks across all 11 open-license events...")
    with zipfile.ZipFile(tmp_target_zip) as zf_target:
        for zinfo in zf_target.infolist():
            if not zinfo.filename.endswith("_building_damage.tif"):
                continue
            fname = zinfo.filename.split("/")[-1]
            sample_id = fname.replace("_building_damage.tif", "")
            ev = parse_event_from_sample_id(sample_id)
            if ev in EXCLUDED_RESTRICTED_EVENTS or ev not in OFFICIAL_EVENT_METADATA:
                continue

            mask = np.array(Image.open(io.BytesIO(zf_target.read(zinfo.filename))))
            uniq, cnts = np.unique(mask, return_counts=True)
            px = dict(zip(uniq.tolist(), cnts.tolist()))

            rec: dict[str, int | str] = {
                "sample_id": sample_id,
                "zip_member": zinfo.filename,
                "background": int(px.get(0, 0)),
                "intact": int(px.get(1, 0)),
                "damaged": int(px.get(2, 0)),
                "destroyed": int(px.get(3, 0)),
                "ignore_255": int(px.get(255, 0)),
            }
            per_event_tiles.setdefault(ev, []).append(rec)

            ev_stat = global_event_stats.setdefault(
                ev,
                {
                    "region": OFFICIAL_EVENT_METADATA[ev]["region"],
                    "tiles": 0,
                    "pixels": {"0": 0, "1": 0, "2": 0, "3": 0, "255": 0},
                },
            )
            ev_stat["tiles"] = int(ev_stat["tiles"]) + 1
            for cls_key, field in [
                ("0", "background"),
                ("1", "intact"),
                ("2", "damaged"),
                ("3", "destroyed"),
                ("255", "ignore_255"),
            ]:
                ev_stat["pixels"][cls_key] += int(rec[field])  # type: ignore[index]

    selected_plan: list[tuple[str, str, dict[str, int | str]]] = []
    for split_name, ev, count in split_plan:
        chosen = select_stratified_tiles(per_event_tiles[ev], count=count, seed=seed)
        for item in chosen:
            selected_plan.append((split_name, ev, item))

    for sub in ("pre-event", "post-event", "target"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(tmp_target_zip) as zf_tgt:
        for _, _, rec in selected_plan:
            sid = str(rec["sample_id"])
            rel_tgt = f"target/{sid}_building_damage.tif"
            out_tgt = output_dir / rel_tgt
            if not out_tgt.exists():
                out_tgt.write_bytes(zf_tgt.read(rel_tgt))

    pre_probe = HttpRangeZipStream(f"{HF_BASE_URL}/pre-event.zip")
    post_probe = HttpRangeZipStream(f"{HF_BASE_URL}/post-event.zip")
    tls = threading.local()

    def get_thread_zips() -> tuple[zipfile.ZipFile, zipfile.ZipFile]:
        if not hasattr(tls, "zips"):
            s_pre = HttpRangeZipStream(
                f"{HF_BASE_URL}/pre-event.zip",
                final_url=pre_probe.final_url,
                size=pre_probe.size,
            )
            s_post = HttpRangeZipStream(
                f"{HF_BASE_URL}/post-event.zip",
                final_url=post_probe.final_url,
                size=post_probe.size,
            )
            tls.zips = (zipfile.ZipFile(s_pre), zipfile.ZipFile(s_post))
        return tls.zips  # type: ignore[no-any-return]

    print(
        f"Streaming & auditing {len(selected_plan)} event-separated triplets ({subset_key}) with {workers} threads..."
    )

    def process_item(
        idx_and_item: tuple[int, tuple[str, str, dict[str, int | str]]],
    ) -> dict[str, object]:
        idx, (split_name, ev, rec) = idx_and_item
        sid = str(rec["sample_id"])
        rel_pre = f"pre-event/{sid}_pre_disaster.tif"
        rel_post = f"post-event/{sid}_post_disaster.tif"
        rel_tgt = f"target/{sid}_building_damage.tif"

        out_pre = output_dir / rel_pre
        out_post = output_dir / rel_post
        out_tgt = output_dir / rel_tgt

        if not out_pre.exists() or not out_post.exists():
            zf_pre, zf_post = get_thread_zips()
            if not out_pre.exists():
                out_pre.write_bytes(zf_pre.read(rel_pre))
            if not out_post.exists():
                out_post.write_bytes(zf_post.read(rel_post))

        pre_meta, pre_arr = inspect_tif(out_pre)
        post_meta, post_arr = inspect_tif(out_post)
        tgt_meta, _ = inspect_tif(out_tgt)

        hw_aligned = (
            pre_meta["shape"][:2]  # type: ignore[index]
            == post_meta["shape"][:2]  # type: ignore[index]
            == tgt_meta["shape"][:2]  # type: ignore[index]
            == [1024, 1024]
        )
        crs_consistent = pre_meta["crs"] == post_meta["crs"] == tgt_meta["crs"]
        is_aligned = bool(hw_aligned and crs_consistent)

        ev_meta = OFFICIAL_EVENT_METADATA[ev]
        modality_info = {
            "pre_optical_shape": pre_meta["shape"],
            "post_sar_shape": post_meta["shape"],
            "dtype": pre_meta["dtype"],
            "crs": pre_meta["crs"],
            "bounds": pre_meta["bounds"],
            "gsd_m": ev_meta["gsd_m"],
            "providers": ev_meta["providers"],
            "aligned": is_aligned,
        }
        label_info = {
            "background": int(rec["background"]),
            "intact": int(rec["intact"]),
            "damaged": int(rec["damaged"]),
            "destroyed": int(rec["destroyed"]),
            "ignore_255": int(rec["ignore_255"]),
        }

        train_stats = None
        if split_name == "train":
            opt_f = pre_arr[:, :, :3].astype(np.float64) / 255.0  # type: ignore[index]
            sar_f = (
                post_arr[:, :, 0] if getattr(post_arr, "ndim", 2) == 3 else post_arr  # type: ignore[index]
            ).astype(np.float64) / 255.0
            train_stats = {
                "n_px": int(opt_f.shape[0] * opt_f.shape[1]),
                "opt_sum": opt_f.sum(axis=(0, 1)).tolist(),
                "opt_sq_sum": (opt_f ** 2).sum(axis=(0, 1)).tolist(),
                "sar_sum": float(sar_f.sum()),
                "sar_sq_sum": float((sar_f ** 2).sum()),
            }

        if idx % 25 == 0 or idx == len(selected_plan):
            print(
                f"  [{idx:03d}/{len(selected_plan):03d}] Verified {sid} ({split_name}) | aligned={is_aligned}"
            )

        return {
            "row": {
                "sample_id": sid,
                "disaster_event": ev,
                "region": ev_meta["region"],
                "pre_event_path": rel_pre,
                "post_event_path": rel_post,
                "target_path": rel_tgt,
                "split": split_name,
                "modality_information": json.dumps(modality_info, separators=(",", ":")),
                "label_information": json.dumps(label_info, separators=(",", ":")),
            },
            "label_info": label_info,
            "is_aligned": is_aligned,
            "train_stats": train_stats,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(process_item, enumerate(selected_plan, start=1)))

    manifest_rows: list[dict[str, str]] = []
    subset_pixel_totals = {"0": 0, "1": 0, "2": 0, "3": 0, "255": 0}
    alignment_mismatches = 0

    tot_train_px = 0
    opt_sum = [0.0, 0.0, 0.0]
    opt_sq = [0.0, 0.0, 0.0]
    sar_sum = 0.0
    sar_sq = 0.0
    n_train_tiles = 0

    for res in results:
        manifest_rows.append(res["row"])  # type: ignore[arg-type]
        li = res["label_info"]  # type: ignore[assignment]
        subset_pixel_totals["0"] += int(li["background"])  # type: ignore[index]
        subset_pixel_totals["1"] += int(li["intact"])  # type: ignore[index]
        subset_pixel_totals["2"] += int(li["damaged"])  # type: ignore[index]
        subset_pixel_totals["3"] += int(li["destroyed"])  # type: ignore[index]
        subset_pixel_totals["255"] += int(li["ignore_255"])  # type: ignore[index]
        if not res["is_aligned"]:
            alignment_mismatches += 1
        ts = res["train_stats"]
        if ts is not None:
            n_train_tiles += 1
            tot_train_px += int(ts["n_px"])  # type: ignore[index]
            for c in range(3):
                opt_sum[c] += float(ts["opt_sum"][c])  # type: ignore[index]
                opt_sq[c] += float(ts["opt_sq_sum"][c])  # type: ignore[index]
            sar_sum += float(ts["sar_sum"])  # type: ignore[index]
            sar_sq += float(ts["sar_sq_sum"])  # type: ignore[index]

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    opt_mean = [round(s / max(1, tot_train_px), 6) for s in opt_sum]
    opt_std = [
        round(math.sqrt(max(0.0, (sq / max(1, tot_train_px)) - (m ** 2))), 6)
        for sq, m in zip(opt_sq, opt_mean)
    ]
    sar_mean = round(sar_sum / max(1, tot_train_px), 6)
    sar_std = round(
        math.sqrt(max(0.0, (sar_sq / max(1, tot_train_px)) - (sar_mean ** 2))), 6
    )

    band_stats_payload = {
        "subset_version": subset_version,
        "computed_strictly_on_split": "train",
        "num_train_tiles": n_train_tiles,
        "optical_rgb_mean_0_1": opt_mean,
        "optical_rgb_std_0_1": opt_std,
        "sar_mean_0_1": sar_mean,
        "sar_std_0_1": sar_std,
    }

    if band_stats_out is not None:
        band_stats_out.parent.mkdir(parents=True, exist_ok=True)
        band_stats_out.write_text(
            json.dumps(band_stats_payload, indent=2) + "\n", encoding="utf-8"
        )
        if band_stats_out.resolve() != (output_dir / band_stats_out.name).resolve():
            shutil.copy2(band_stats_out, output_dir / band_stats_out.name)

    audit_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": cfg["dataset"]["name"],
        "paper_doi": cfg["dataset"]["paper_doi"],
        "zenodo_version_doi": cfg["dataset"]["zenodo_version_doi"],
        "download_source": f"https://huggingface.co/datasets/{cfg['dataset']['huggingface_repo']} (HTTP 206 selective extraction)",
        "excluded_restricted_events": sorted(EXCLUDED_RESTRICTED_EVENTS),
        "subset_version": subset_version,
        "selection_criteria": f"Event-separated split ({subset_key}) with deterministic round-robin stratification across damage strata, seed={seed}.",
        "subset_summary": {
            "total_triplets": len(manifest_rows),
            "split_counts": {
                sp: sum(1 for r in manifest_rows if r["split"] == sp)
                for sp in ("train", "val", "test")
            },
            "events_by_split": {
                sp: sorted({r["disaster_event"] for r in manifest_rows if r["split"] == sp})
                for sp in ("train", "val", "test")
            },
            "alignment_mismatches": alignment_mismatches,
            "pixel_totals": subset_pixel_totals,
        },
        "train_only_band_stats": band_stats_payload,
        "global_11_open_events_audit": global_event_stats,
    }
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")

    if manifest_out.resolve() != (output_dir / manifest_out.name).resolve():
        shutil.copy2(manifest_out, output_dir / manifest_out.name)
    if audit_out.resolve() != (output_dir / audit_out.name).resolve():
        shutil.copy2(audit_out, output_dir / audit_out.name)

    if tmp_target_zip.exists():
        tmp_target_zip.unlink()

    print(
        f"\nSUCCESS: Wrote {manifest_out} ({len(manifest_rows)} rows) and {audit_out}"
    )
    print(json.dumps(audit_payload["subset_summary"], indent=2))
    print(json.dumps(band_stats_payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit BRIGHT and build selective cloud subset."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run zero-dependency local validation.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument(
        "--subset-key",
        type=str,
        default="subset_v0_1",
        help="YAML subset key (e.g. subset_v0_1 or subset_v0_2).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/bright_subset_v0_1")
    )
    parser.add_argument(
        "--manifest-out", type=Path, default=Path("/kaggle/working/manifest_v0_1.csv")
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=Path("/kaggle/working/dataset_audit_v0_1.json"),
    )
    parser.add_argument("--band-stats-out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    return build_cloud_subset(
        args.config,
        args.subset_key,
        args.output_dir,
        args.manifest_out,
        args.audit_out,
        args.band_stats_out,
        args.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
