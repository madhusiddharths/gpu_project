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

import numpy as np
import warp as wp

#: Vector field type for position, velocity, force.
vec3 = wp.vec3f

#: Scalar field type for radius, mass, inverse mass, timestep.
scalar = wp.float32

#: Matching host-side dtype.
np_scalar = np.float32

#: Machine epsilon for the working precision.
EPS = float(np.finfo(np_scalar).eps)


def accumulation_bound(max_coord: float, steps: int) -> float:
    """Worst-case position drift from repeated addition, in metres.

    Each add of a small increment to a coordinate of magnitude |x| rounds by at
    most 0.5 * ulp(x) = 0.5 * EPS * |x|. Summing that over `steps` additions and
    taking the largest coordinate reached gives a rigorous upper bound. It is
    loose by roughly 1-2 orders of magnitude in practice because not every
    rounding is worst-case and not all share a sign, but it is a bound, so a
    test written against it cannot fail spuriously.

    Used by the integrator tests and, from Block 17, by the overlap diagnostics.
    """
    return 0.5 * EPS * abs(max_coord) * steps