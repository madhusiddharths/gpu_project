"""Axis-aligned box boundaries — six rigid planes.

A wall contact is a particle contact whose partner has infinite mass, infinite
radius and zero velocity. Every consequence follows from those three facts:

    R* = r          a plane has infinite radius, so 1/R* = 1/r + 0
    m* = m          the reduced mass of a body and an immovable one is the body
    E* = E/(1-nu^2) a rigid wall adds no compliance to the series sum, so the
    G* = G/(2-nu)   effective moduli are exactly TWICE the two-sphere values

The stiffness doubling is not a detail. Wall contacts are the stiffest in the
domain, so they are what actually sets the timestep in a bounded run, and a
Rayleigh bound computed only from particle-particle contact is optimistic.

NO ATOMICS ANYWHERE IN THIS MODULE. A wall is not a body, so there is no partner
to deposit an equal and opposite force onto — the only thing a wall kernel
writes is particle i's own force, torque and history, and thread i is the sole
writer of all three. Particle-particle contact cannot have this property, and
the contrast is the clearest illustration in the project of why boundary
conditions are cheap and interactions are not.

TANGENTIAL HISTORY IS TRIVIAL HERE. Slot w of particle i IS wall w — the
identity of the partner is the slot index. There is nothing to match, nothing
to carry forward, and no double buffer: a contact that ends simply has its slot
zeroed. Compare history.py, which needs an entire re-keying pass to achieve the
same thing for pairs whose identity is not knowable from an index.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects.
"""

import numpy as np
import warp as wp

from warp_dem.forces import (
    hertz_normal_force,
    rotate_into_tangent_plane,
    tangential_stiffness,
)
from warp_dem.materials import ContactParams
from warp_dem.precision import np_scalar, scalar, vec3
from warp_dem.state import ParticleState

#: Walls are indexed 0-5. Walls 0-2 are the LOW faces on the x, y, z axes;
#: walls 3-5 are the HIGH faces. `axis = w % 3` recovers the axis and `w < 3`
#: the side, which is what lets one loop cover all six.
WALL_COUNT = 6

WALL_NAMES = ("-x", "-y", "-z", "+x", "+y", "+z")


@wp.kernel
def accumulate_wall_forces(
    pos: wp.array(dtype=vec3),
    vel: wp.array(dtype=vec3),
    omega: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    mass: wp.array(dtype=scalar),
    force: wp.array(dtype=vec3),
    torque: wp.array(dtype=vec3),
    wall_disp: wp.array(dtype=vec3),
    bounds_min: vec3,
    bounds_max: vec3,
    dt: scalar,
    params: ContactParams,
):
    """Hertz-Mindlin contact against six planes. One thread per particle.

    The six walls are handled by one loop with a fixed trip count, so every
    thread does identical work and the warp never diverges on wall COUNT — only
    on which walls are actually touched, and in a real bed almost every particle
    touches none or one.
    """
    i = wp.tid()

    xi = pos[i]
    vi = vel[i]
    wi = omega[i]
    ri = radius[i]
    mi = mass[i]

    for w in range(WALL_COUNT):
        axis = w % 3
        n = vec3(0.0, 0.0, 0.0)
        dist = scalar(0.0)

        # n points from the particle centre TOWARD the wall, matching the
        # particle-particle convention where the normal points from i to j.
        if w < 3:
            n[axis] = -1.0
            dist = xi[axis] - bounds_min[axis]
        else:
            n[axis] = 1.0
            dist = bounds_max[axis] - xi[axis]

        slot = i * WALL_COUNT + w
        delta = ri - dist

        if delta > 0.0 and dist > -ri:
            lever = (ri - 0.5 * delta) * n

            # The wall is stationary, so the relative surface velocity is
            # entirely the particle's own.
            v_rel = vi + wp.cross(wi, lever)
            v_n = wp.dot(v_rel, n)
            v_t = v_rel - v_n * n

            # A plane has infinite radius and infinite mass.
            f_n = hertz_normal_force(delta, ri, mi, v_n, params)

            disp = rotate_into_tangent_plane(wall_disp[slot], n)
            disp = disp + v_t * dt

            s_t = tangential_stiffness(params.g_eff, ri, delta)
            c_t = 2.0 * wp.sqrt(5.0 / 6.0) * (-params.beta) * wp.sqrt(s_t * mi)
            damping = c_t * v_t
            f_t = -s_t * disp - damping

            limit = params.mu_s * wp.abs(f_n)
            magnitude = wp.length(f_t)
            if magnitude > limit:
                if magnitude > 1.0e-20:
                    f_t = f_t * (limit / magnitude)
                disp = -(f_t + damping) / s_t

            wall_disp[slot] = disp

            tau_r = vec3(0.0, 0.0, 0.0)
            if params.mu_r > 0.0:
                w_mag = wp.length(wi)
                if w_mag > 1.0e-12:
                    tau_r = -(params.mu_r * wp.abs(f_n) * ri / w_mag) * wi

            # No atomic: thread i is the only writer of particle i.
            force[i] = force[i] + f_t - f_n * n
            torque[i] = torque[i] + wp.cross(lever, f_t) + tau_r
        else:
            # Contact ended. Zeroing the slot IS the eviction — a wall's
            # identity is its index, so there is nothing else to clean up.
            wall_disp[slot] = vec3(0.0, 0.0, 0.0)


class BoxBoundary:
    """Six axis-aligned planes bounding the domain.

    Phase 1's boundary condition. The drum in Phase 5 replaces it, but the
    contact law does not change — only the geometry that computes `delta` and
    the normal.
    """

    def __init__(
        self,
        state: ParticleState,
        params: ContactParams,
        bounds_min,
        bounds_max,
    ):
        lo = np.asarray(bounds_min, dtype=np_scalar)
        hi = np.asarray(bounds_max, dtype=np_scalar)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError(f"bounds must be 3-vectors, got {lo.shape} and {hi.shape}")
        if np.any(hi <= lo):
            raise ValueError(f"bounds_max must exceed bounds_min, got {lo} and {hi}")

        self.params = params
        self.bounds_min = vec3(*(float(v) for v in lo))
        self.bounds_max = vec3(*(float(v) for v in hi))
        self.disp = wp.zeros(state.n * WALL_COUNT, dtype=vec3, device=state.device)
        self.n = state.n

    @classmethod
    def from_config(cls, state: ParticleState, params: ContactParams, cfg):
        """Build from a `configs/geometry/box.yaml` node."""
        return cls(state, params, cfg.bounds_min, cfg.bounds_max)

    def apply(self, state: ParticleState, dt: float) -> None:
        wp.launch(
            accumulate_wall_forces,
            dim=state.n,
            inputs=[
                state.pos,
                state.vel,
                state.omega,
                state.radius,
                state.mass,
                state.force,
                state.torque,
                self.disp,
                self.bounds_min,
                self.bounds_max,
                float(dt),
                self.params,
            ],
            device=state.device,
        )

    def displacements(self) -> np.ndarray:
        """Device-to-host read, shape (n, 6, 3). Diagnostics and tests only."""
        return self.disp.numpy().reshape(self.n, WALL_COUNT, 3)
