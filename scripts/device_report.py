"""Capture device and environment provenance.

Every benchmark number this project reports must be traceable to the machine,
driver, library versions, and exact commit that produced it. A throughput
figure without provenance is not a result, it is a rumour.

Usage:
    python scripts/device_report.py
    python scripts/device_report.py --out results/provenance.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import warp as wp

from warp_dem import __version__ as warp_dem_version
from warp_dem import resolve_device


def _shell(cmd: list[str]) -> str | None:
    """Run a command, return stripped stdout, or None if it isn't available."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def git_state() -> dict:
    """Exact commit, and whether the tree was dirty when this ran."""
    commit = _shell(["git", "rev-parse", "HEAD"])
    dirty = _shell(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(dirty) if dirty is not None else None,
        "branch": _shell(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def gpu_state() -> list[dict]:
    """Per-device info from Warp, plus driver version from nvidia-smi."""
    devices = []
    for d in wp.get_cuda_devices():
        devices.append(
            {
                "name": d.name,
                "arch": f"sm_{d.arch}",
                "total_memory_gb": round(d.total_memory / 1e9, 2),
                "is_uva": d.is_uva,
            }
        )

    driver = _shell(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    for d in devices:
        d["driver_version"] = driver

    return devices


def package_versions() -> dict:
    """Versions of everything that could move a number."""
    pkgs = {"warp-lang": wp.config.version, "warp-dem": warp_dem_version}
    for name in ("numpy", "cupy", "cudf", "scipy"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkgs[name] = None
    return pkgs


def collect() -> dict:
    wp.init()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": sys.version.split()[0],
        },
        "git": git_state(),
        "packages": package_versions(),
        "warp_devices": [str(d) for d in wp.get_devices()],
        "cuda_devices": gpu_state(),
        "default_device": str(resolve_device("auto")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = collect()
    text = json.dumps(report, indent=2)
    print(text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nwritten to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()