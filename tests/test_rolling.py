"""Rolling resistance torque — Block 14.

Rolling resistance is the least first-principles term in the contact model. A
perfect sphere on a perfect surface touches at a point, and a point exerts no
couple, so an ideal sphere rolls forever. Real grains have facets, asperities
and a finite contact patch over which normal pressure piles up on the leading
edge when the grain rolls, producing a couple that opposes the rotation.

    tau_r = -mu_r |F_n| R_eff * omega_rel / |omega_rel|

CONSTANT DIRECTIONAL TORQUE. The magnitude is independent of rotation rate; only
the direction depends on omega. That is what makes it a FRICTION rather than a
viscosity, and it is why a pile can stand: a velocity-proportional model exerts
nothing at rest and so cannot arrest creep.

mu_r is CALIBRATED, not measured. docs/validation_targets.md is explicit that
adopting 0.09 as an input and then "validating" against the repose angle it was
fitted to would be circular. These tests therefore assert the MECHANICS of the
term — magnitude, direction, rate-independence — and never that a particular
mu_r produces a particular angle. That claim belongs to Phase 3, and only via
Targets 2 and 3, which were not used to fit it.
"""

import dataclasses

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.forces import ContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget

RADIUS = 0.002
DENSITY = 2500.0
NO_GRAVITY = (0.0, 0.0, 0.0)

BASE = MaterialProperties(
    name="test_glass",
    density=DENSITY,
    radius=RADIUS,
    youngs_modulus=1.0e8,
    poisson_ratio=0.20,
    restitution=0.90,
    friction_particle=0.0,  # isolate rolling from sliding friction
    friction_wall=0.0,
    rolling_friction=0.05,
    restitution_wall=0.90,
)


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _budget(material):
    return compute_budget(
        material.radius, material.density, material.youngs_modulus,
        material.poisson_ratio, impact_velocity=1.0, hertz_steps=50,
    )


def _pressed_pair(device, material, omega_i, overlap_ratio=1e-3, pinned=(0, 1)):
    """Two overlapping particles, both pinned, particle 0 spinning.

    Pinning both fixes the geometry exactly, so the measured torque is a clean
    function of a known overlap and a known relative spin. A free pair would fly
    apart under the pre-loaded overlap before anything could be read — the
    mistake that produced a zero-force reading in Block 13.
    """
    r = material.radius
    overlap = overlap_ratio * r
    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[0.0, 0.0, 0.0], [2.0 * r - overlap, 0.0, 0.0]])
    state.set_angular_velocities([omega_i, [0.0, 0.0, 0.0]])
    if pinned:
        state.set_fixed(list(pinned))
    model = ContactModel(state, material.pair_params())
    solver = Solver(state, dt=_budget(material).limit, gravity=NO_GRAVITY,
                    budget=_budget(material), contact_model=model)
    return state, solver, overlap


def _expected_torque(material, overlap):
    """mu_r |F_n| R_eff, with F_n the static Hertz force at this overlap."""
    params = material.pair_params()
    r_eff = material.radius / 2.0
    f_n = (4.0 / 3.0) * params.e_eff * np.sqrt(r_eff) * overlap**1.5
    return material.rolling_friction * f_n * r_eff


# ── mechanics of the term ─────────────────────────────────────────────────────


def test_no_rolling_torque_without_relative_rotation(device):
    """A contact that is not rotating has nothing to resist."""
    state, _, _ = _pressed_pair(device, BASE, omega_i=[0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.torques(), 0.0, atol=1e-15)


def test_no_rolling_torque_when_the_coefficient_is_zero(device):
    material = dataclasses.replace(BASE, rolling_friction=0.0)
    state, _, _ = _pressed_pair(device, material, omega_i=[0.0, 0.0, 30.0])
    np.testing.assert_allclose(state.torques(), 0.0, atol=1e-15)


def test_rolling_torque_opposes_relative_rotation(device):
    """The torque is antiparallel to omega_rel, whatever axis it lies on."""
    for axis in ([0.0, 0.0, 30.0], [0.0, 25.0, 0.0], [12.0, -8.0, 5.0]):
        state, _, _ = _pressed_pair(device, BASE, omega_i=axis)
        torque = state.torques()[0]
        omega = np.array(axis)
        cosine = float(np.dot(torque, omega)) / (
            np.linalg.norm(torque) * np.linalg.norm(omega)
        )
        assert cosine == pytest.approx(-1.0, abs=1e-4), f"axis {axis}: cos {cosine}"


def test_rolling_torque_magnitude_matches_mu_r_times_normal_force(device):
    """|tau_r| = mu_r |F_n| R_eff, checked against the analytic Hertz force."""
    state, _, overlap = _pressed_pair(device, BASE, omega_i=[0.0, 0.0, 30.0])
    measured = float(np.linalg.norm(state.torques()[0]))
    assert measured == pytest.approx(_expected_torque(BASE, overlap), rel=0.01)


def test_rolling_torque_is_independent_of_rotation_rate(device):
    """The defining property: a FRICTION, not a viscosity.

    A rate-proportional model would give torques in the ratio 1 : 10 : 100 here.
    Constant directional torque gives 1 : 1 : 1, which is what lets a bed come
    to rest instead of creeping indefinitely.
    """
    magnitudes = []
    for rate in (3.0, 30.0, 300.0):
        state, _, _ = _pressed_pair(device, BASE, omega_i=[0.0, 0.0, rate])
        magnitudes.append(float(np.linalg.norm(state.torques()[0])))

    assert magnitudes[1] == pytest.approx(magnitudes[0], rel=1e-4)
    assert magnitudes[2] == pytest.approx(magnitudes[0], rel=1e-4)


def test_rolling_torque_scales_with_normal_load(device):
    """Deeper contact, larger F_n, proportionally larger resistance."""
    magnitudes = {}
    for ratio in (1e-3, 4e-3):
        state, _, overlap = _pressed_pair(
            device, BASE, omega_i=[0.0, 0.0, 30.0], overlap_ratio=ratio
        )
        magnitudes[ratio] = float(np.linalg.norm(state.torques()[0]))

    # F_n ~ delta^1.5, so four times the overlap is 8x the force and torque.
    assert magnitudes[4e-3] / magnitudes[1e-3] == pytest.approx(8.0, rel=0.02)


@pytest.mark.parametrize("mu_r", [0.01, 0.05, 0.2])
def test_rolling_torque_is_linear_in_the_coefficient(device, mu_r):
    material = dataclasses.replace(BASE, rolling_friction=mu_r)
    state, _, overlap = _pressed_pair(device, material, omega_i=[0.0, 0.0, 30.0])
    measured = float(np.linalg.norm(state.torques()[0]))
    assert measured == pytest.approx(_expected_torque(material, overlap), rel=0.01)


def test_rolling_torque_is_equal_and_opposite_on_the_two_partners(device):
    """A couple, not a force: it must not change the total angular momentum
    of the pair, only exchange it between them."""
    state, _, _ = _pressed_pair(device, BASE, omega_i=[0.0, 0.0, 30.0])
    torques = state.torques()
    np.testing.assert_allclose(torques[0], -torques[1], rtol=1e-5)


# ── rolling decay ─────────────────────────────────────────────────────────────


def test_spin_decays_and_stops_rather_than_reversing(device):
    """A spinning particle held against another must halt, not oscillate.

    The danger with a constant-magnitude resisting torque is overshoot: once
    omega reaches zero the torque should vanish, but a naive implementation keeps
    applying full torque and drives the spin negative, so the particle
    oscillates forever. The guard on |omega_rel| is what prevents that, and this
    test is what proves the guard works.
    """
    material = dataclasses.replace(BASE, rolling_friction=0.2)
    # Particle 1 pinned as the surface; particle 0 free to spin down. Its
    # inv_mass stays finite so it can move, but with no gravity and a
    # frictionless normal contact it stays put.
    r = material.radius
    overlap = 1e-3 * r
    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[0.0, 0.0, 0.0], [2.0 * r - overlap, 0.0, 0.0]])
    state.set_angular_velocities([[0.0, 0.0, 20.0], [0.0, 0.0, 0.0]])
    state.set_fixed([1])
    budget = _budget(material)
    solver = Solver(state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
                    contact_model=ContactModel(state, material.pair_params()))

    history = []
    for _ in range(60):
        solver.run(20)
        history.append(float(state.angular_velocities()[0][2]))

    assert history[0] < 20.0, "spin did not decay at all"
    assert min(history) > -1.0, f"spin reversed — torque overshot zero: {history}"
    # Monotone decay while the contact lasts.
    decaying = [a >= b - 1e-6 for a, b in zip(history, history[1:], strict=False)]
    assert all(decaying), f"spin did not decay monotonically: {history}"
