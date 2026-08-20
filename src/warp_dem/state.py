"""Particle state, stored as structure-of-arrays.

One Warp array per physical field. This layout is chosen for memory coalescing
on the GPU: a warp of 32 threads reading `radius[i]` touches contiguous memory,
whereas the same read from an array-of-structs would scatter across ~32 memory
sectors. DEM is memory-bound, so that ratio is close to a runtime ratio.

SoA is used from the first commit because retrofitting it later means touching
every kernel signature, every test, and the entire Phase 6 analytics layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_dem.precision import np_scalar, quat, scalar, vec3


@dataclass
class ParticleState:
    """Structure-of-arrays container for particle degrees of freedom.

    Every field is a separate device array of length ``n``. Nothing here keeps a
    host-side copy: ``.numpy()`` on any field is an explicit device-to-host
    transfer and must never appear inside a simulation loop.

    Inertia is stored as a SCALAR, not a tensor. That is exact for a sphere,
    whose mass distribution is identical about every axis, and it is what makes
    world-frame angular velocity valid here: for an isotropic body the
    gyroscopic coupling term (omega x I.omega) vanishes identically.

    PHASE 4 WILL BREAK THIS. A glued-sphere tablet has a genuine inertia
    tensor, constant only in the body frame, and requires Euler's equations
    with the gyroscopic term plus a world<->body transform each step.
    """

    n: int
    device: wp.context.Device

    pos: wp.array       # wp.vec3f  [m]
    vel: wp.array       # wp.vec3f  [m/s]
    force: wp.array     # wp.vec3f  [N]

    radius: wp.array    # float32   [m]
    mass: wp.array      # float32   [kg]
    inv_mass: wp.array  # float32   [1/kg]

    orient: wp.array       # wp.quatf, unit, body -> world
    omega: wp.array        # wp.vec3f, rad/s, WORLD frame
    torque: wp.array       # wp.vec3f, N.m, world frame

    inertia: wp.array      # float32, kg.m^2  -- SCALAR, isotropic
    inv_inertia: wp.array  # float32, 1/(kg.m^2)

    @classmethod
    def allocate(
        cls,
        n: int,
        device: wp.context.Device,
        radius: float,
        density: float,
    ) -> ParticleState:
        """Allocate a monodisperse population at the origin, at rest.

        ``inv_mass`` is stored rather than computed in-kernel: division is
        markedly more expensive than multiplication on GPU hardware, and it
        would otherwise be recomputed for every particle on every one of the
        two integrator kernels, every step. It also gives a clean way to pin a
        particle later — inv_mass = 0 is infinite mass.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if radius <= 0.0 or density <= 0.0:
            raise ValueError(f"radius and density must be positive, got {radius}, {density}")

        r = np.full(n, radius, dtype=np_scalar)
        volume = (4.0 / 3.0) * np.pi * r.astype(np.float64) ** 3
        m = (density * volume).astype(np_scalar)

        moment = (0.4 * m * r**2).astype(np_scalar)  # (2/5) m r^2, solid sphere

        identity = np.zeros((n, 4), dtype=np_scalar)
        identity[:, 3] = 1.0  # (x, y, z, w): w = 1 is the identity rotation
        return cls(
            n=n,
            device=device,
            pos=wp.zeros(n, dtype=vec3, device=device),
            vel=wp.zeros(n, dtype=vec3, device=device),
            force=wp.zeros(n, dtype=vec3, device=device),
            radius=wp.array(r, dtype=scalar, device=device),
            mass=wp.array(m, dtype=scalar, device=device),
            inv_mass=wp.array((1.0 / m).astype(np_scalar), dtype=scalar, device=device),
            orient=wp.array(identity, dtype=quat, device=device),
            omega=wp.zeros(n, dtype=vec3, device=device),
            torque=wp.zeros(n, dtype=vec3, device=device),
            inertia=wp.array(moment, dtype=scalar, device=device),
            inv_inertia=wp.array((1.0 / moment).astype(np_scalar), dtype=scalar, device=device),
        )

    def set_positions(self, values) -> None:
        """Host-to-device write. Setup only — never inside a loop."""
        self.pos.assign(self._as_vec3_array(values))

    def set_velocities(self, values) -> None:
        """Host-to-device write. Setup only — never inside a loop."""
        self.vel.assign(self._as_vec3_array(values))

    def _as_vec3_array(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (self.n, 3):
            raise ValueError(f"expected shape ({self.n}, 3), got {arr.shape}")
        return arr

    def positions(self) -> np.ndarray:
        """Device-to-host read. Diagnostics and tests only."""
        return self.pos.numpy()

    def velocities(self) -> np.ndarray:
        """Device-to-host read. Diagnostics and tests only."""
        return self.vel.numpy()

    def forces(self) -> np.ndarray:
        """Device-to-host read. Diagnostics and tests only."""
        return self.force.numpy()

    def set_angular_velocities(self, values) -> None:
        """Host-to-device write. Setup only — never inside a loop."""
        self.omega.assign(self._as_vec3_array(values))

    def set_orientations(self, values) -> None:
        """Host-to-device write. Input is normalised; near-zero input is rejected.

        Silently normalising is the right call: users supply axis-angle results
        or hand-written quaternions that are unit to five digits, and refusing
        those would be pedantry. A zero-length quaternion is a real error.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (self.n, 4):
            raise ValueError(f"expected shape ({self.n}, 4), got {arr.shape}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("quaternion with near-zero length has no orientation")
        self.orient.assign((arr / norms).astype(np_scalar))

    def orientations(self) -> np.ndarray:
        """Device-to-host read. Diagnostics and tests only."""
        return self.orient.numpy()

    def angular_velocities(self) -> np.ndarray:
        """Device-to-host read. Diagnostics and tests only."""
        return self.omega.numpy()

    def torques(self) -> np.ndarray:
        """Device-to-host read. Diagnostics and tests only."""
        return self.torque.numpy()