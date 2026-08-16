"""warp-dem: a GPU-native discrete element solver for granular coating processes."""

__version__ = "0.1.0"

from warp_dem.device import DeviceError, assert_same_device, describe_device, resolve_device

__all__ = [
    "resolve_device",
    "describe_device",
    "assert_same_device",
    "DeviceError",
    "__version__",
]