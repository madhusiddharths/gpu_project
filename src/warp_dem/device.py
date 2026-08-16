"""Device resolution — the single point of truth for CPU vs CUDA selection.

NOTHING else in this codebase may hardcode a device string. Every module that
needs a device receives one as an argument, resolved here from config.

This exists because the project's core benchmark claim is "same code, same
machine, one flag different." A stray hardcoded device anywhere silently
invalidates that claim.
"""

from __future__ import annotations

import warp as wp


class DeviceError(RuntimeError):
    """Raised when a requested device is unavailable or malformed."""


def resolve_device(name: str = "auto") -> "wp.context.Device":
    """Turn a config string into a Warp device object.

    Args:
        name: One of "auto", "cpu", "cuda", "cuda:0", "cuda:1", ...
              "auto" picks the first CUDA device if any exist, else CPU.

    Returns:
        A Warp device object suitable for passing to wp.array / wp.launch.

    Raises:
        DeviceError: if a CUDA device was explicitly requested but none exists,
                     or the name is not recognised.
    """
    wp.init()

    key = (name or "auto").strip().lower()

    if key == "auto":
        cuda_devices = wp.get_cuda_devices()
        if cuda_devices:
            return cuda_devices[0]
        return wp.get_device("cpu")

    if key.startswith("cuda"):
        cuda_devices = wp.get_cuda_devices()
        if not cuda_devices:
            raise DeviceError(
                f"Device '{name}' requested but no CUDA device is available. "
                f"This build reports devices: {[str(d) for d in wp.get_devices()]}. "
                "On Apple Silicon this is expected — use device=cpu locally, "
                "or run on the remote GPU instance."
            )

    try:
        return wp.get_device(key)
    except Exception as exc:
        raise DeviceError(
            f"Unrecognised device '{name}'. "
            f"Available: {[str(d) for d in wp.get_devices()]}"
        ) from exc


def describe_device(device: "wp.context.Device") -> str:
    """One-line human-readable summary, for logs and benchmark provenance."""
    if device.is_cuda:
        return (
            f"CUDA device '{device.name}' | "
            f"arch sm_{device.arch} | "
            f"total memory {device.total_memory / 1e9:.1f} GB"
        )
    return f"CPU device '{device.name}'"


def assert_same_device(*arrays: wp.array) -> "wp.context.Device":
    """Guard: all arrays must live on one device. Returns that device.

    Cross-device array access is a class of bug that produces either a crash
    or, worse, a silent wrong answer. Call this at the top of any function
    that takes multiple arrays.
    """
    devices = {str(a.device) for a in arrays if a is not None}
    if len(devices) > 1:
        raise DeviceError(f"Arrays span multiple devices: {sorted(devices)}")
    if not devices:
        raise DeviceError("No arrays supplied.")
    return arrays[0].device