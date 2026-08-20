"""Velocity-Verlet integration and body-force kernels.

Kick-drift-kick form:

    v_{n+1/2} = v_n + (dt/2) * a_n          <- kick_drift, first half
    x_{n+1}   = x_n + dt * v_{n+1/2}        <- kick_drift, drift
    a_{n+1}   = f(x_{n+1}) / m              <- force kernels, between launches
    v_{n+1}   = v_{n+1/2} + (dt/2) * a_{n+1} <- kick

Symplectic, second-order, one force evaluation per step. Under constant
acceleration it is exact, which is what makes the free-fall test a sharp check
on plumbing rather than a check on integration accuracy.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects; stringified annotations break it.
"""

import warp as wp


@wp.kernel
def apply_body_force(
    force: wp.array(dtype=wp.vec3f),
    mass: wp.array(dtype=wp.float32),
    gravity: wp.vec3f,
):
    """Overwrite the force array with the body force.

    This kernel ASSIGNS rather than accumulates, so it doubles as the per-step
    zeroing pass and must run FIRST in every force evaluation. Every contact
    kernel added later accumulates into this array with wp.atomic_add, because
    a particle receives contributions from many contacts, resolved by different
    threads, simultaneously.
    """
    i = wp.tid()
    force[i] = gravity * mass[i]


@wp.kernel
def kick_drift(
    pos: wp.array(dtype=wp.vec3f),
    vel: wp.array(dtype=wp.vec3f),
    force: wp.array(dtype=wp.vec3f),
    inv_mass: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    """First half-kick using the OLD force, then the full-step drift."""
    i = wp.tid()
    accel = force[i] * inv_mass[i]
    v_half = vel[i] + accel * (0.5 * dt)
    vel[i] = v_half
    pos[i] = pos[i] + v_half * dt


@wp.kernel
def kick(
    vel: wp.array(dtype=wp.vec3f),
    force: wp.array(dtype=wp.vec3f),
    inv_mass: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    """Second half-kick using the NEW force."""
    i = wp.tid()
    accel = force[i] * inv_mass[i]
    vel[i] = vel[i] + accel * (0.5 * dt)