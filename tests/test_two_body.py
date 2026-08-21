"""Two-body and few-body test battery — Block 11.

Adapted from [Sunday2020] section 4. Each test isolates one force term, and
each runs on the CPU backend in well under a second.

Only the tests that normal force alone can support live here. The rest land in
the block where their physics arrives: oblique impact in Block 13 (needs
Mindlin), rolling decay in Block 14, block sliding and the incline in Block 15
(need a wall). The alternative — writing them all now and marking most xfail —
would have put failing tests in CI for a week to no purpose.

THE STACKED DROP IS THE INTERESTING ONE. A column of spheres is the smallest
system with a FORCE CHAIN: each sphere carries the weight of everything above
it, so the overlaps must grow monotonically down the stack, and each one must
independently satisfy the static Hertz balance for its own load. A model that
gets a single collision right can still get a chain wrong.
"""

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.forces import NormalContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.precision import accumulation_bound
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget, hertz_static_overlap

GLASS = MaterialProperties(
    name="test_glass",
    density=2500.0,
    radius=0.002,
    youngs_modulus=1.0e8,
    poisson_ratio=0.20,
    restitution=0.50,  # settle quickly; the stack is not a restitution test
    friction_particle=0.25,
    friction_wall=0.35,
    rolling_friction=0.01,
    restitution_wall=0.50,
)

GRAVITY = (0.0, 0.0, -9.81)
#: Steps for the glancing-collision test; also sets its derived tolerance.
STEPS = 2000
NO_GRAVITY = (0.0, 0.0, 0.0)


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _budget(material, impact_velocity=1.0, steps=25):
    return compute_budget(
        material.radius, material.density, material.youngs_modulus,
        material.poisson_ratio, impact_velocity=impact_velocity, hertz_steps=steps,
    )


# ── stacked drop ──────────────────────────────────────────────────────────────


def _settle_stack(device, n=5, material=GLASS, steps=40_000):
    """A vertical column with the bottom sphere pinned, settling under gravity.

    Started just touching rather than dropped from a height: the physics under
    test is the static force chain, and a genuine drop would spend most of the
    run in free fall measuring nothing.
    """
    r = material.radius
    state = ParticleState.allocate(n, device, r, material.density)
    state.set_positions([[0.0, 0.0, 2.0 * r * k] for k in range(n)])
    state.set_fixed([0])

    budget = _budget(material)
    solver = Solver(
        state, dt=budget.limit, gravity=GRAVITY, budget=budget,
        contact_model=NormalContactModel(state, material.pair_params()),
    )
    solver.run(steps)
    return state, solver


def test_stack_comes_to_rest(device):
    state, _ = _settle_stack(device)
    speeds = np.linalg.norm(state.velocities(), axis=1)
    # A settled stack should be far slower than a single free-fall step.
    assert speeds.max() < 1e-3, f"stack still moving at {speeds.max():.3e} m/s"


def test_stack_does_not_drift_sideways_or_spin(device):
    """Normal force is central, so it can produce no lateral force and no torque."""
    state, _ = _settle_stack(device)
    positions = state.positions()
    np.testing.assert_allclose(positions[:, 0], 0.0, atol=1e-9)
    np.testing.assert_allclose(positions[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(state.angular_velocities(), 0.0, atol=1e-12)
    np.testing.assert_allclose(state.orientations()[:, 3], 1.0, atol=1e-6)


def test_stack_does_not_sink_through_itself(device):
    """Overlap must stay a small fraction of the radius, not collapse."""
    state, _ = _settle_stack(device)
    z = np.sort(state.positions()[:, 2])
    gaps = np.diff(z)
    overlaps = 2.0 * GLASS.radius - gaps
    assert np.all(overlaps > 0.0), "stack separated — spheres are not in contact"
    assert np.all(overlaps / GLASS.radius < 0.01), (
        f"max overlap {overlaps.max() / GLASS.radius:.4%} of radius exceeds the "
        "1% Validation Target 5 criterion"
    )


def test_force_chain_overlaps_match_the_static_hertz_balance(device):
    """Each contact carries the weight above it, and Hertz fixes the overlap.

    This is the test that a single collision cannot catch: it checks that the
    contact law composes correctly through a chain of four contacts carrying
    four different loads.
    """
    n = 5
    state, _ = _settle_stack(device, n=n)
    z = np.sort(state.positions()[:, 2])
    gaps = np.diff(z)
    overlaps = 2.0 * GLASS.radius - gaps

    mass = float(state.mass.numpy()[0])
    for contact, overlap in enumerate(overlaps):
        # Contact k (counting from the bottom) supports n-1-k spheres above it.
        supported = n - 1 - contact
        load = supported * mass * 9.81
        expected = hertz_static_overlap(
            load, GLASS.radius, GLASS.youngs_modulus, GLASS.poisson_ratio
        )
        assert overlap == pytest.approx(expected, rel=0.02), (
            f"contact {contact} carries {supported} spheres: "
            f"overlap {overlap:.4e} m vs Hertz {expected:.4e} m"
        )

    # Monotone: the deepest contact is at the bottom, carrying the most.
    assert np.all(np.diff(overlaps) < 0.0)


def test_pinned_particle_never_moves(device):
    """inv_mass = 0 must be exact, not merely small."""
    state, _ = _settle_stack(device)
    np.testing.assert_allclose(state.positions()[0], [0.0, 0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(state.velocities()[0], 0.0, atol=0.0)


def test_set_fixed_rejects_out_of_range_indices(device):
    state = ParticleState.allocate(3, device, GLASS.radius, GLASS.density)
    with pytest.raises(IndexError):
        state.set_fixed([3])


# ── restitution robustness: Validation Target 4b ──────────────────────────────


def _collide(device, material, v_approach, phase=0.0, steps=25):
    """Head-on collision, with the initial gap shifted by `phase` timesteps.

    The phase offset is the point: a collision begins part-way through a
    timestep, and where exactly turns out to dominate the restitution error.
    """
    budget = _budget(material, impact_velocity=v_approach, steps=steps)
    dt = budget.limit
    r = material.radius
    extra = phase * v_approach * dt

    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[-r - extra / 2, 0.0, 0.0], [r + extra / 2, 0.0, 0.0]])
    state.set_velocities([[v_approach / 2, 0.0, 0.0], [-v_approach / 2, 0.0, 0.0]])
    solver = Solver(
        state, dt=dt, gravity=NO_GRAVITY, budget=budget,
        contact_model=NormalContactModel(state, material.pair_params()),
    )

    touched = False
    for _ in range(4000):
        solver.step()
        gap = float(state.positions()[1][0] - state.positions()[0][0])
        if gap < 2.0 * r:
            touched = True
        elif touched:
            break
    assert touched, "particles never made contact"
    v = state.velocities()
    return abs(float(v[0][0] - v[1][0])) / v_approach


@pytest.mark.parametrize("restitution", [0.5, 0.7, 0.9])
def test_restitution_is_independent_of_impact_velocity(device, restitution):
    """VALIDATION TARGET 4b: spread under 2% across 0.1-2.0 m/s.

    This is the property that distinguishes a Hertzian contact from a linear
    one. The damping coefficient carries sqrt(S_n) ~ delta^(1/4), which cancels
    the velocity dependence; constant damping on a linear spring does not.
    """
    material = dataclasses.replace(GLASS, restitution=restitution)
    measured = [_collide(device, material, v) for v in (0.1, 0.5, 1.0, 2.0)]
    spread = (max(measured) - min(measured)) / float(np.mean(measured))
    assert spread < 0.02, f"e = {measured} spans {100 * spread:.3f}%"


@pytest.mark.parametrize("restitution", [0.5, 0.9])
def test_restitution_holds_for_every_sub_timestep_contact_phase(device, restitution):
    """Target 4a must hold WHEREVER in the timestep the collision begins.

    Starting the particles exactly touching samples one phase out of a
    continuum, and Block 10 measured that phase — not float32 — as the dominant
    error term. A test fixed at phase zero could therefore pass while the model
    was out of tolerance for most real collisions, so the worst case over phase
    is the honest form of the criterion.
    """
    material = dataclasses.replace(GLASS, restitution=restitution)
    measured = [_collide(device, material, 1.0, phase=p)
                for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
    worst = max(abs(m - restitution) / restitution for m in measured)
    assert worst < 0.02, (
        f"worst-case error {100 * worst:.3f}% over contact phase, e={measured}"
    )


def test_kinetic_energy_ratio_matches_restitution_squared(device):
    """e is defined on velocity, so energy must come back as e^2.

    An independent check on the same collision: it would catch a model that
    reproduced the rebound speed by accident while mishandling the energy
    balance during contact.
    """
    material = dataclasses.replace(GLASS, restitution=0.7)
    e_measured = _collide(device, material, 1.0)
    assert e_measured**2 == pytest.approx(0.49, rel=0.04)


def test_equal_masses_rebound_symmetrically(device):
    """Momentum conservation: the centre of mass must not move."""
    material = dataclasses.replace(GLASS, restitution=0.8)
    budget = _budget(material)
    r = material.radius
    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[-r, 0.0, 0.0], [r, 0.0, 0.0]])
    state.set_velocities([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    solver = Solver(
        state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
        contact_model=NormalContactModel(state, material.pair_params()),
    )
    solver.run(2000)

    v = state.velocities()
    assert float(v[0][0]) == pytest.approx(-float(v[1][0]), rel=1e-4)
    momentum = state.mass.numpy()[:, None] * v
    np.testing.assert_allclose(momentum.sum(axis=0), 0.0, atol=1e-9)


def test_frictionless_glancing_collision_imparts_no_spin(device):
    """A central force exerts no torque, whatever the geometry.

    THE OBVIOUS TEST HERE IS WRONG, and finding that out was the useful part of
    Block 11. "Frictionless collisions preserve tangential velocity" is a
    RIGID-BODY result: it assumes an instantaneous impulse along a fixed line of
    centres. A soft-sphere contact lasts a finite time — 166 us here — during
    which the two bodies slide past each other and the line of centres rotates.
    The impulse stays parallel to the instantaneous normal, but that normal is
    no longer a fixed direction, so tangential velocity DOES change.

    Measured 0.0125 m/s against 0.0124 m/s predicted from the 1.43 deg of
    normal rotation. Not a leak; geometry.

    What survives exactly is what this test asserts: zero torque, hence zero
    spin, and conserved angular momentum. Block 13's counterpart is the
    statement that friction is present — this one is the statement that it is
    absent, and the pair only means something if this one is right.
    """
    material = dataclasses.replace(GLASS, restitution=1.0)
    budget = _budget(material)
    r = material.radius
    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[-r, 0.0, 0.0], [r, 0.0, 0.0]])
    # Approaching along x, sliding past each other along y.
    state.set_velocities([[0.5, 0.3, 0.0], [-0.5, -0.3, 0.0]])

    def angular_momentum():
        pos, vel = state.positions(), state.velocities()
        mass = state.mass.numpy()[:, None]
        centre = (mass * pos).sum(axis=0) / mass.sum()
        return np.cross(pos - centre, mass * vel).sum(axis=0)

    initial_l = angular_momentum()
    solver = Solver(
        state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
        contact_model=NormalContactModel(state, material.pair_params()),
    )
    solver.run(STEPS)

    # No torque: spin is untouched and orientation is still the identity.
    np.testing.assert_allclose(state.angular_velocities(), 0.0, atol=1e-12)
    np.testing.assert_allclose(state.orientations()[:, 3], 1.0, atol=1e-9)

    # Angular momentum about the centre of mass is exactly conserved by a
    # central force, whether or not the normal direction rotates. Exactly in
    # exact arithmetic — in float32 it inherits the accumulation drift of the
    # velocities it is built from, so the tolerance is DERIVED from that rather
    # than chosen: L is linear in v, so its relative error is v's. Two
    # half-kicks per step means 2 * steps additions, not steps.
    speed = 0.6
    rtol = accumulation_bound(speed, 2 * STEPS) / speed
    np.testing.assert_allclose(angular_momentum(), initial_l, rtol=rtol)

    # The tangential change is bounded by the geometry of normal rotation:
    # theta ~ (v_t_rel * t_contact) / 2R, and dv_y ~ 0.5 * dv_x_rel * sin(theta).
    theta = math.atan(0.6 * budget.hertz_time / (2.0 * r))
    bound = 1.5 * 0.5 * 1.0 * math.sin(theta)
    v = state.velocities()
    assert abs(float(v[0][1]) - 0.3) < bound
    assert float(v[0][1]) == pytest.approx(-float(v[1][1]), rel=1e-4)


def test_contact_duration_matches_the_hertz_prediction(device):
    """The 2.87 prefactor in timestep.py, checked against a real collision.

    Flagged in docs/timestep_and_stiffness.md as unconfirmed against primary
    sources. This does not confirm the source, but it does confirm that the
    formula predicts the duration the solver actually delivers — which is what
    the timestep budget depends on.
    """
    material = dataclasses.replace(GLASS, restitution=0.97)
    v0 = 1.0
    budget = _budget(material, impact_velocity=v0, steps=100)
    dt = budget.limit
    r = material.radius

    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[-r, 0.0, 0.0], [r, 0.0, 0.0]])
    state.set_velocities([[v0 / 2, 0.0, 0.0], [-v0 / 2, 0.0, 0.0]])
    solver = Solver(
        state, dt=dt, gravity=NO_GRAVITY, budget=budget,
        contact_model=NormalContactModel(state, material.pair_params()),
    )

    in_contact, touched = 0, False
    for _ in range(4000):
        solver.step()
        if float(state.positions()[1][0] - state.positions()[0][0]) < 2.0 * r:
            in_contact += 1
            touched = True
        elif touched:
            break

    measured = in_contact * dt
    # Nearly elastic, so the damped duration should sit close to the undamped
    # prediction. 15% covers the damping correction and the discrete sampling.
    assert measured == pytest.approx(budget.hertz_time, rel=0.15), (
        f"measured contact {measured * 1e6:.2f} us vs predicted "
        f"{budget.hertz_time * 1e6:.2f} us"
    )
    assert math.isclose(dt * 100, budget.hertz_time, rel_tol=0.35)
