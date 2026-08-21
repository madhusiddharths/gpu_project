"""Initial particle placement.

One function, and an API shape chosen because of a bug that occurred twice
during Block 17: a lattice was generated first and a domain was hard-coded
second, the lattice overran the domain, and the outermost particles started
buried several millimetres inside a wall. Hertz force goes as delta^1.5, so a
5 mm overlap on a 2 mm particle is enormous, and the bed launched itself at
12 m/s. Nothing crashed — the run simply measured an explosion.

So the bounds are an INPUT here, not an output. The lattice is fitted inside
them and the function raises if the requested particles cannot fit. It is not
possible to use this helper and end up with a particle inside a wall.

There is no random close packing here, deliberately. A jittered cubic lattice
dropped under gravity settles into a realistic packing within a few thousand
steps, and the settling is physics the solver should be doing rather than
something the initialiser should fake.
"""

from __future__ import annotations

import numpy as np

from warp_dem.precision import np_scalar


def lattice_fill(
    n: int,
    radius: float,
    bounds_min,
    bounds_max,
    spacing_factor: float = 2.2,
    jitter: float = 0.05,
    seed: int = 0,
    drop_height: float = 0.0,
) -> np.ndarray:
    """Place `n` particles on a jittered cubic lattice inside the domain.

    Args:
        radius: particle radius. Every centre is kept at least this far from
            every wall, so no particle starts in contact with the boundary.
        spacing_factor: lattice pitch in units of radius. Must exceed
            2 + 2*jitter or jittered neighbours could overlap on creation.
        jitter: random displacement as a fraction of radius, breaking the
            lattice symmetry so the bed does not settle into an unphysically
            crystalline packing.
        drop_height: raises the whole lattice by this many radii, so the bed
            settles under gravity instead of starting in contact.

    Returns:
        Array of shape (n, 3) in the working precision.

    Raises:
        ValueError: if `n` particles cannot fit inside the usable volume.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if spacing_factor <= 2.0 + 2.0 * jitter:
        raise ValueError(
            f"spacing_factor {spacing_factor} is too tight for jitter {jitter}: "
            f"jittered neighbours could overlap on creation. Use more than "
            f"{2.0 + 2.0 * jitter}."
        )

    lo = np.asarray(bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max, dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError(f"bounds must be 3-vectors, got {lo.shape} and {hi.shape}")

    # A centre must stay one radius clear of every face, and the jitter has to
    # fit inside that clearance too.
    margin = radius * (1.0 + jitter)
    usable = (hi - margin) - (lo + margin)
    if np.any(usable <= 0.0):
        raise ValueError(
            f"domain {lo} to {hi} is too small to hold a particle of radius {radius}"
        )

    pitch = spacing_factor * radius
    counts = np.floor(usable / pitch).astype(int) + 1
    capacity = int(np.prod(counts))
    if capacity < n:
        raise ValueError(
            f"cannot fit {n} particles of radius {radius} in domain {lo} to {hi} "
            f"at spacing {spacing_factor} R: capacity is {capacity} "
            f"({counts[0]} x {counts[1]} x {counts[2]}).\n"
            f"Either enlarge the domain, reduce run.n_particles, or reduce "
            f"run.spacing_factor (but keep it above {2.0 + 2.0 * jitter})."
        )

    # Fill along z last so the lattice grows upward: a partially filled top
    # layer settles harmlessly, whereas a partially filled column would leave
    # a gap in the middle of the bed.
    grid = np.array(
        [
            (i, j, k)
            for k in range(counts[2])
            for j in range(counts[1])
            for i in range(counts[0])
        ],
        dtype=np.float64,
    )[:n]

    positions = lo + margin + grid * pitch
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        positions += rng.uniform(-jitter * radius, jitter * radius, size=positions.shape)
    positions[:, 2] += drop_height * radius

    # Assert the postcondition the whole module exists to guarantee.
    if np.any(positions < lo + radius) or np.any(positions > hi - radius):
        raise ValueError(
            "internal error: generated packing does not clear the domain walls"
        )
    return positions.astype(np_scalar)
