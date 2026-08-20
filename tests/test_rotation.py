"""Rotational integration tests.

Comparisons are made by ROTATING A TEST VECTOR, never by comparing quaternion
components. Unit quaternions double-cover the rotation group: q and -q describe
the same physical orientation, so a componentwise comparison can report a
mismatch where none exists.

As in test_integrator.py, accuracy tests use a coarse timestep. The exponential
map is exact for constant angular velocity, so a finer dt adds only float32
accumulation noise. The one fine-dt test here checks NORM STABILITY, not
accuracy, which is exactly what a long run stresses.
"""

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.integrate import kick
from warp_dem.solver import Solver
from warp_dem.state import ParticleState

RADIUS = 0.002
DENSITY = 2500.0
NO_GRAVITY = (0.0, 0.0, 0.0)

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _rotate(q, v):
    """Rotate v by unit quaternion q = (x, y, z, w), in float64 on the host.

    An independent reference implementation. Using Warp's own quat_rotate here
    would make the test agree with the code by construction rather than by
    correctness.
    """
    u = np.asarray(q[:3], dtype=np.float64)
    w = float(q[3])
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _spin(device, omega, dt, steps, q0=None):
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    if q0 is not None:
        state.set_orientations([q0])
    state.set_angular_velocities([omega])
    Solver(state, dt=dt, gravity=NO_GRAVITY).run(steps)
    return state


def test_inertia_from_geometry(device):
    """Solid sphere: I = (2/5) m r^2, and the stored inverse is consistent."""
    state = ParticleState.allocate(3, device, RADIUS, DENSITY)
    expected = 0.4 * state.mass.numpy() * RADIUS**2
    np.testing.assert_allclose(state.inertia.numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(
        state.inertia.numpy() * state.inv_inertia.numpy(), 1.0, rtol=1e-5
    )


def test_allocation_is_identity_orientation(device):
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    np.testing.assert_allclose(state.orientations(), [[0, 0, 0, 1]] * 2, atol=1e-7)
    for q in state.orientations():
        np.testing.assert_allclose(_rotate(q, X_HAT), X_HAT, atol=1e-7)


def test_zero_angular_velocity_leaves_orientation_fixed(device):
    """Exercises the near-zero guard in advance_orientation. Without it, the
    axis normalisation divides by zero and every orientation becomes NaN."""
    state = _spin(device, [0.0, 0.0, 0.0], dt=1e-4, steps=1000)
    q = state.orientations()[0]
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(_rotate(q, X_HAT), X_HAT, atol=1e-7)


def test_quarter_turn_about_z_is_right_handed(device):
    """omega along +z must carry +x toward +y. A sign error here inverts every
    rotation in the project and is invisible in a norm check."""
    omega_z = np.pi / 2.0  # rad/s
    state = _spin(device, [0.0, 0.0, omega_z], dt=1e-3, steps=1000)  # t = 1 s
    np.testing.assert_allclose(_rotate(state.orientations()[0], X_HAT), Y_HAT, atol=1e-5)


def test_free_rotation_matches_analytic(device):
    """Constant angular velocity about a tilted axis, compared against a
    directly constructed axis-angle rotation. The exponential map is exact
    here, so the residual is pure float32 roundoff."""
    axis = np.array([1.0, -2.0, 3.0])
    axis /= np.linalg.norm(axis)
    rate = 7.0  # rad/s
    dt, steps = 1e-3, 1500
    t = dt * steps

    state = _spin(device, (axis * rate).tolist(), dt=dt, steps=steps)

    theta = rate * t
    q_exact = np.concatenate([axis * np.sin(theta / 2.0), [np.cos(theta / 2.0)]])

    for v in (X_HAT, Y_HAT, Z_HAT):
        np.testing.assert_allclose(
            _rotate(state.orientations()[0], v), _rotate(q_exact, v), atol=1e-5
        )


def test_full_revolution_returns_to_start(device):
    """2*pi radians about z in 1 s returns every body axis to where it began —
    even though the quaternion itself has flipped sign (double cover). This is
    the test that would fail if orientations were compared componentwise."""
    state = _spin(device, [0.0, 0.0, 2.0 * np.pi], dt=1e-3, steps=1000)
    q = state.orientations()[0]

    for v in (X_HAT, Y_HAT, Z_HAT):
        np.testing.assert_allclose(_rotate(q, v), v, atol=1e-5)

    assert q[3] < 0.0, "expected the double-cover sign flip after one revolution"


def test_quaternion_stays_normalised_over_long_run(device):
    """The Block 7 done-when criterion. 100k steps of tumbling about a tilted
    axis; the norm must not creep. Checks stability, not accuracy."""
    steps = 100_000
    state = _spin(device, [3.0, -4.0, 5.0], dt=1e-5, steps=steps)
    q = state.orientations()[0]

    assert np.all(np.isfinite(q))
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6


def test_orientation_is_history_dependent_not_additive(device):
    """Rotation composition does not commute, so the integrator must compose
    rather than accumulate. Two 90-degree turns applied in opposite orders must
    give different results; if they matched, the update would be adding
    something vector-like and would be wrong."""
    quarter = np.pi / 2.0
    dt, steps = 1e-3, 1000  # 1 s per leg

    def two_legs(first, second):
        state = ParticleState.allocate(1, device, RADIUS, DENSITY)
        state.set_angular_velocities([(np.array(first) * quarter).tolist()])
        Solver(state, dt=dt, gravity=NO_GRAVITY).run(steps)
        q_mid = state.orientations()[0]

        state2 = ParticleState.allocate(1, device, RADIUS, DENSITY)
        state2.set_orientations([q_mid.tolist()])
        state2.set_angular_velocities([(np.array(second) * quarter).tolist()])
        Solver(state2, dt=dt, gravity=NO_GRAVITY).run(steps)
        return state2.orientations()[0]

    zx = _rotate(two_legs(Z_HAT, X_HAT), Y_HAT)
    xz = _rotate(two_legs(X_HAT, Z_HAT), Y_HAT)
    assert np.linalg.norm(zx - xz) > 0.5


def test_angular_kick_applies_torque(device):
    """Torque -> angular acceleration, tested at the kernel level.

    Nothing in the solver produces torque yet (gravity acts at the centre of
    mass), so the kick kernel is launched directly with a filled torque array
    rather than adding a body-torque hook to the Solver that physical runs
    would never use.
    """
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    tau_z = 1.0e-9
    state.torque.assign(np.array([[0.0, 0.0, tau_z]], dtype=np.float32))

    dt = 1e-4
    wp.launch(
        kick,
        dim=1,
        inputs=[
            state.vel, state.force, state.inv_mass,
            state.omega, state.torque, state.inv_inertia,
            dt,
        ],
        device=device,
    )

    inv_i = float(state.inv_inertia.numpy()[0])
    np.testing.assert_allclose(
        state.angular_velocities()[0], [0.0, 0.0, 0.5 * dt * inv_i * tau_z], rtol=1e-4
    )


def test_translation_and_rotation_do_not_interfere(device):
    """The fused kick_drift kernel touches position, velocity, orientation and
    angular velocity in one launch. A misplaced index would couple them.

    Non-interference is a claim about two RUNS, not about analytic accuracy, so
    the comparison is against the same simulation with the other degree of
    freedom switched off. Both runs execute identical arithmetic on identical
    data, so they must agree BIT FOR BIT — the shared float32 roundoff cancels
    instead of masking the signal, and there is no tolerance to tune. Comparing
    against the closed-form trajectory instead would fold in ~5e-5 m of
    accumulation drift and hide any coupling smaller than that.
    """
    dt, steps = 1e-3, 1000
    spin = [0.0, 0.0, np.pi / 2.0]
    gravity = (0.0, 0.0, -9.81)

    def simulate(with_spin: bool, with_gravity: bool):
        state = ParticleState.allocate(1, device, RADIUS, DENSITY)
        state.set_positions([[0.0, 0.0, 1.0]])
        if with_spin:
            state.set_angular_velocities([spin])
        Solver(state, dt=dt, gravity=gravity if with_gravity else NO_GRAVITY).run(steps)
        return state

    both = simulate(with_spin=True, with_gravity=True)
    falling_only = simulate(with_spin=False, with_gravity=True)
    spinning_only = simulate(with_spin=True, with_gravity=False)

    # Spinning must not perturb the fall, to the last bit.
    np.testing.assert_array_equal(both.positions(), falling_only.positions())
    np.testing.assert_array_equal(both.velocities(), falling_only.velocities())

    # Falling must not perturb the spin, to the last bit.
    np.testing.assert_array_equal(both.orientations(), spinning_only.orientations())
    np.testing.assert_array_equal(
        both.angular_velocities(), spinning_only.angular_velocities()
    )

    # Sanity: both degrees of freedom actually did something, so the equalities
    # above are not trivially satisfied by nothing having moved.
    assert float(both.positions()[0, 2]) < 0.0
    np.testing.assert_allclose(_rotate(both.orientations()[0], X_HAT), Y_HAT, atol=1e-5)

def test_rejects_degenerate_orientation(device):
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    with pytest.raises(ValueError):
        state.set_orientations([[0.0, 0.0, 0.0, 0.0]])