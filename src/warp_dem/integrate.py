"""Velocity-Verlet integration, translational and rotational.

Kick-drift-kick, applied to both degrees of freedom:

    v_{n+1/2} = v_n + (dt/2) * f_n / m          <- kick_drift
    w_{n+1/2} = w_n + (dt/2) * tau_n / I
    x_{n+1}   = x_n + dt * v_{n+1/2}
    q_{n+1}   = exp(w_{n+1/2} * dt) . q_n
    f_{n+1}, tau_{n+1} = forces(x_{n+1}, q_{n+1})   <- between launches
    v_{n+1}   = v_{n+1/2} + (dt/2) * f_{n+1} / m <- kick
    w_{n+1}   = w_{n+1/2} + (dt/2) * tau_{n+1} / I

SCOPE: inertia is a scalar. Exact for spheres, whose mass distribution is
isotropic, which is also why world-frame angular velocity is valid here — the
gyroscopic term (omega x I.omega) vanishes for an isotropic body. Phase 4's
glued-sphere particles need a tensor, body-frame integration, and Euler's
equations with that term restored.

FUSION: translation and rotation share one kernel rather than two. They touch
disjoint arrays, so there is no data-reuse benefit; the reason is launch
overhead. Two kernels per half-step instead of one is 4 launches per timestep,
and at a few microseconds each across tens of millions of steps that is real
wall-clock for zero benefit.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects.
"""

import warp as wp

from warp_dem.precision import quat, scalar, vec3


@wp.func
def advance_orientation(q: quat, omega_world: vec3, dt: scalar) -> quat:
    """Compose the rotation actually performed this step onto the orientation.

    Exponential map: during dt the body turns through |omega|*dt radians about
    the axis omega/|omega|. Building that increment from an axis and an angle
    makes it unit-norm by construction, so unlike a linearised update the
    renormalisation below is sweeping up float32 roundoff rather than repairing
    a method that emits invalid quaternions. It is also exact for constant
    omega, which is what makes the free-rotation test sharp.

    omega is in the WORLD frame, so the increment composes on the LEFT.

    The guard exists because a zero angular velocity has no rotation axis to
    normalise. It is a divergent branch, but in a real bed nearly every
    particle is rotating, so warps are close to uniform on it.
    """
    angle = wp.length(omega_world) * dt
    if angle > 1.0e-12:
        dq = wp.quat_from_axis_angle(wp.normalize(omega_world), angle)
        return wp.normalize(wp.mul(dq, q))
    return q


@wp.kernel
def initialise_forces(
    force: wp.array(dtype=vec3),
    torque: wp.array(dtype=vec3),
    mass: wp.array(dtype=scalar),
    gravity: vec3,
):
    """Overwrite force with the body force and clear torque.

    ASSIGNS rather than accumulates, so it doubles as the per-step zeroing pass
    and MUST run first in every force evaluation. Every contact kernel added
    later accumulates with wp.atomic_add, because one particle receives
    contributions from many contacts resolved by different threads at once.

    Gravity acts at the centre of mass, so it contributes no torque.
    """
    i = wp.tid()
    force[i] = gravity * mass[i]
    torque[i] = vec3(0.0, 0.0, 0.0)


@wp.kernel
def kick_drift(
    pos: wp.array(dtype=vec3),
    vel: wp.array(dtype=vec3),
    force: wp.array(dtype=vec3),
    inv_mass: wp.array(dtype=scalar),
    orient: wp.array(dtype=quat),
    omega: wp.array(dtype=vec3),
    torque: wp.array(dtype=vec3),
    inv_inertia: wp.array(dtype=scalar),
    dt: scalar,
):
    """First half-kick on both velocities using the OLD force and torque,
    then the full-step drift of position and orientation."""
    i = wp.tid()
    half = 0.5 * dt

    v_half = vel[i] + force[i] * inv_mass[i] * half
    vel[i] = v_half
    pos[i] = pos[i] + v_half * dt

    w_half = omega[i] + torque[i] * inv_inertia[i] * half
    omega[i] = w_half
    orient[i] = advance_orientation(orient[i], w_half, dt)


@wp.kernel
def kick(
    vel: wp.array(dtype=vec3),
    force: wp.array(dtype=vec3),
    inv_mass: wp.array(dtype=scalar),
    omega: wp.array(dtype=vec3),
    torque: wp.array(dtype=vec3),
    inv_inertia: wp.array(dtype=scalar),
    dt: scalar,
):
    """Second half-kick using the NEW force and torque."""
    i = wp.tid()
    half = 0.5 * dt
    vel[i] = vel[i] + force[i] * inv_mass[i] * half
    omega[i] = omega[i] + torque[i] * inv_inertia[i] * half