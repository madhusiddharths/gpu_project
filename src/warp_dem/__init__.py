"""GPU-native discrete element solver for granular coating processes.

Phase 1 surface. Import the solver pieces from their own modules
(`warp_dem.solver`, `warp_dem.forces`, ...) — only the small, widely used
helpers are re-exported here.
"""

__version__ = "0.1.0"

from warp_dem.device import DeviceError, assert_same_device, describe_device, resolve_device
from warp_dem.materials import (
    ContactParams,
    MaterialProperties,
    damping_ratio,
    effective_shear_modulus,
    wall_shear_modulus,
    wall_youngs_modulus,
)
from warp_dem.precision import (
    EPS,
    PRECISION,
    accumulation_bound,
    cancellation_bound,
)
from warp_dem.timestep import (
    TimestepBudget,
    TimestepError,
    assert_timestep_valid,
    compute_budget,
    hertz_static_overlap,
)

__all__ = [
    # device
    "resolve_device",
    "describe_device",
    "assert_same_device",
    "DeviceError",
    # timestep
    "TimestepBudget",
    "TimestepError",
    "assert_timestep_valid",
    "compute_budget",
    "hertz_static_overlap",
    # materials
    "ContactParams",
    "MaterialProperties",
    "damping_ratio",
    "effective_shear_modulus",
    "wall_youngs_modulus",
    "wall_shear_modulus",
    # precision
    "PRECISION",
    "EPS",
    "accumulation_bound",
    "cancellation_bound",
    "__version__",
]
