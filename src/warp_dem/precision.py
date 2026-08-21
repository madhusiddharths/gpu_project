"""Floating-point precision policy — one decision, one place.

Positions, velocities and forces are float32. That is the norm for GPU DEM and
it is where the performance story lives: DEM is memory-bound, so float64 would
double the bytes moved for every particle field on every kernel.

The measured cost, characterised in Block 6: float32 position accumulation
drifts. Each `x += v*dt` rounds by up to 0.5*ulp(x), and for a near-constant
increment those roundings are CORRELATED rather than random, so they accumulate
roughly linearly in step count instead of as sqrt(N). See docs/precision.md.

Every kernel annotation and every array allocation in this package uses the
aliases below. Flipping the project to double precision is then an edit here,
not a search-and-replace across the solver — the same reasoning that put device
resolution in exactly one module.

NOTE: no `from __future__ import annotations` needed here, but modules that
import these into Warp kernel signatures must not use it.
"""

import os

import numpy as np
import warp as wp

#: Working precision, selected by the WARP_DEM_PRECISION environment variable.
#:
#: float32 is the default and the project's decision. The override exists so
#: that the SAME code can be run in double precision without editing a file,
#: which is what makes two things practical:
#:
#:   - the Block 17 drift measurement, which needs a float64 leg as its
#:     reference and runs it as a subprocess
#:   - the Phase 7 mixed-precision experiment the plan already anticipates
#:
#: It is an environment variable rather than a config key on purpose. Types are
#: resolved at IMPORT time, because Warp reads the annotation objects when it
#: generates kernel code, so this cannot be a runtime setting without
#: re-importing the world. Anything decided before `import warp_dem` completes
#: cannot come from Hydra.
PRECISION = os.environ.get("WARP_DEM_PRECISION", "float32").strip().lower()

if PRECISION == "float32":
    #: Vector field type for position, velocity, force.
    vec3 = wp.vec3f
    #: Scalar field type for radius, mass, inverse mass, timestep.
    scalar = wp.float32
    #: Orientation type. Warp quaternions are ordered (x, y, z, w) with the
    #: SCALAR COMPONENT LAST. Roughly half the libraries in existence put it
    #: first; mixing the two produces rotations wrong in ways that look almost
    #: right.
    quat = wp.quatf
    #: Matching host-side dtype.
    np_scalar = np.float32
elif PRECISION == "float64":
    vec3 = wp.vec3d
    scalar = wp.float64
    quat = wp.quatd
    np_scalar = np.float64
else:
    raise ValueError(
        f"WARP_DEM_PRECISION must be 'float32' or 'float64', got {PRECISION!r}"
    )

#: Machine epsilon for the working precision.
EPS = float(np.finfo(np_scalar).eps)


def accumulation_bound(magnitude: float, additions: int) -> float:
    """Worst-case drift from repeated addition into a float32 accumulator.

    Each add of a small increment to an accumulator of magnitude |a| rounds by
    at most 0.5 * ulp(a) = 0.5 * EPS * |a|. Summing that over `additions` gives
    a rigorous upper bound. Loose by 1-2 orders in practice because not every
    rounding is worst-case and not all share a sign, but it IS a bound, so a
    test written against it cannot fail spuriously across platforms.

    Args:
        magnitude: the largest value the accumulator reaches, in its own units.
                   NOT the distance travelled — rounding scales with ulp(a),
                   which scales with |a|.
        additions: the number of accumulating adds, which is NOT always the
                   number of timesteps. Velocity-Verlet updates position once
                   per step (the drift) but velocity TWICE (two half-kicks), so
                   a velocity bound takes 2 * steps.

    Used by the integrator tests and, from Block 17, by the overlap diagnostics.
    """
    return 0.5 * EPS * abs(magnitude) * additions


def cancellation_bound(magnitude: float, difference: float, exponent: float = 1.0) -> float:
    """Relative error when a small `difference` is formed from large operands.

    Contact overlap is the motivating case, characterised in Block 10:

        delta = (r_i + r_j) - |x_j - x_i|

    subtracts two numbers of order 4e-3 m to obtain one of order 2e-7 m. The
    operands each carry absolute error up to ulp(magnitude) = EPS * magnitude,
    and subtraction preserves that absolute error while destroying almost all
    of the significant digits, so the RELATIVE error of the result is inflated
    by the ratio magnitude / difference. This is catastrophic cancellation, and
    unlike the accumulation drift in `accumulation_bound` it is not a function
    of how long the simulation has run — it is present on the very first step.

    The `exponent` argument carries the error through a power law. Hertz force
    goes as delta^1.5, so a relative error r in delta becomes 1.5 * r in the
    force. Measured at delta/R = 1e-4: predicted 3.6e-3, observed 1.8e-3.

    Two consequences worth remembering:

    - Precision in the overlap degrades as the overlap SHRINKS. A stiffer
      material gives smaller overlaps and therefore noisier contact forces,
      which is a second, quieter cost of stiffness scaling alongside the
      timestep. It is also why a barely-touching contact is the least
      accurately resolved one in the whole bed.
    - The standard mitigation is the same as for accumulation drift: keep
      coordinates small by storing them relative to a cell or domain origin,
      which shrinks ulp without widening the type. Flagged for Phase 7.

    Args:
        magnitude: scale of the operands being subtracted (here, r_i + r_j).
        difference: the small result (here, the overlap delta).
        exponent: power the difference is subsequently raised to.

    Returns:
        A relative error bound, dimensionless.
    """
    if difference <= 0.0:
        raise ValueError(f"difference must be positive, got {difference}")
    return exponent * EPS * abs(magnitude) / difference