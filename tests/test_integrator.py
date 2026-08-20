"""Velocity-Verlet integrator tests.

Two families, deliberately separated.

TRAJECTORY TESTS use a COARSE timestep. Velocity-Verlet is algebraically exact
under constant acceleration, so a smaller dt buys no accuracy — it only adds
float32 accumulation noise, while the bugs being hunted get quieter. A missing
initial force evaluation shifts the trajectory by roughly 0.5*a*dt*t, which
shrinks linearly with dt. Measured in Block 6, hunting exactly that bug:

    dt=1e-3   signal/noise = 5700
    dt=1e-4   signal/noise = 66
    dt=1e-5   signal/noise = 1.6
    dt=1e-6   bug invisible

Coarse is sharper. This is specific to the constant-acceleration regime; once
contact forces exist the timestep is bounded by Rayleigh and the trade reverses.

PRECISION CHARACTERISATION uses a FINE timestep and asserts the drift stays
inside the derived ulp bound. It documents the float32 behaviour rather than
tuning a tolerance until it disappears. See docs/precision.md.
"""

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.precision import accumulation_bound
from warp_dem.solver import Solver
from warp_dem.state import ParticleState

RADIUS = 0.002
DENSITY = 2500.0
G = (0.0, 0.0, -9.81)


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    """Runs every test below on CPU, and additionally on CUDA when present."""
    return request.param


def _drop(device, x0, v0, dt, steps, gravity=G):
    state = ParticleState.allocate(len(x0), device, RADIUS, DENSITY)
    state.set_positions(x0)
    state.set_velocities(v0)
    Solver(state, dt=dt, gravity=gravity).run(steps)
    return state


def test_mass_from_geometry(device):
    state = ParticleState.allocate(4, device, RADIUS, DENSITY)
    expected = DENSITY * (4.0 / 3.0) * np.pi * RADIUS**3
    np.testing.assert_allclose(state.mass.numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(state.mass.numpy() * state.inv_mass.numpy(), 1.0, rtol=1e-5)


def test_forces_are_computed_before_the_first_step(device):
    """Velocity-Verlet needs f(x_0). A zero force array at step 1 is a silent bug."""
    state = ParticleState.allocate(3, device, RADIUS, DENSITY)
    Solver(state, dt=1e-5, gravity=G)
    expected = state.mass.numpy()[:, None] * np.array(G, dtype=np.float32)
    np.testing.assert_allclose(state.forces(), expected, rtol=1e-5)


def test_free_fall_matches_analytic_trajectory(device):
    """The Block 6 done-when criterion.

    dt = 1e-3, 100 steps. Measured drift here is ~1e-8 m; the tolerance leaves
    roughly 50x margin and still detects a missing-initial-force bug at 4.9e-4 m.
    """
    n, dt, steps = 8, 1e-3, 100
    t = dt * steps

    rng = np.random.default_rng(0)
    x0 = rng.uniform(-0.05, 0.05, size=(n, 3)).astype(np.float32)
    v0 = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)

    state = _drop(device, x0, v0, dt, steps)

    a = np.array(G, dtype=np.float64)
    np.testing.assert_allclose(state.positions(), x0 + v0 * t + 0.5 * a * t**2, atol=1e-6)
    np.testing.assert_allclose(state.velocities(), v0 + a * t, atol=1e-5)


@pytest.mark.parametrize("dt", [1e-3, 1e-4, 1e-5])
def test_exact_under_constant_acceleration_at_every_timestep(device, dt):
    """Truncation error is identically zero; only roundoff remains, so the error
    must sit inside the ulp bound for whatever step count that dt implies."""
    steps = int(round(0.1 / dt))
    state = _drop(device, [[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]], dt, steps)

    z_exact = 1.0 + 0.5 * G[2] * 0.1**2
    assert abs(state.positions()[0, 2] - z_exact) < accumulation_bound(1.0, steps)


def test_coarse_timestep_is_more_accurate_than_fine(device):
    """Counterintuitive and worth pinning: with zero truncation error, a 100x
    finer dt is strictly worse because it does 100x more rounded additions."""
    z_exact = 1.0 + 0.5 * G[2] * 0.1**2

    coarse = _drop(device, [[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]], 1e-3, 100)
    fine = _drop(device, [[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]], 1e-5, 10_000)

    err_coarse = abs(float(coarse.positions()[0, 2]) - z_exact)
    err_fine = abs(float(fine.positions()[0, 2]) - z_exact)
    assert err_coarse < err_fine


def test_float32_drift_stays_within_ulp_bound(device):
    """Characterisation, not a pass/fail on physics.

    Zero gravity, constant velocity, 10k steps of dt=1e-5. Observed drift on
    arm64 is ~8.5e-6 m on the x component — a relative error of 8.5e-5, entirely
    from correlated rounding in `x += v*dt`. Recorded in docs/precision.md;
    revisited as the mixed-precision experiment in Phase 7.
    """
    steps = 10_000
    state = _drop(
        device, [[0.0, 0.0, 0.0]], [[1.0, -2.0, 0.5]], 1e-5, steps, gravity=(0.0, 0.0, 0.0)
    )
    exact = np.array([0.1, -0.2, 0.05])
    err = np.abs(state.positions()[0] - exact)
    assert np.all(err < accumulation_bound(0.2, steps))


def test_all_masses_fall_identically(device):
    """Galileo. Catches mass appearing without its matching inverse."""
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    m = state.mass.numpy()
    m[1] *= 1000.0
    state.mass.assign(m)
    state.inv_mass.assign((1.0 / m).astype(np.float32))

    Solver(state, dt=1e-5, gravity=G).run(1000)
    np.testing.assert_allclose(state.positions()[0], state.positions()[1], atol=1e-9)


def test_zero_gravity_is_straight_line(device):
    state = _drop(
        device, [[0.0, 0.0, 0.0]], [[1.0, -2.0, 0.5]], 1e-3, 100, gravity=(0.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        state.positions()[0], np.array([0.1, -0.2, 0.05]), atol=1e-6
    )


def test_rejects_bad_timestep(device):
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    with pytest.raises(ValueError):
        Solver(state, dt=0.0)