"""Hertz normal force with viscous damping — Block 10.

The headline test is `test_restitution_matches_configured_value`, which is
Validation Target 4a: a Hertz-Mindlin implementation that cannot reproduce its
own input restitution has a damping-coefficient bug and nothing downstream is
worth running.

TOLERANCES ARE DERIVED, NOT PICKED. The 2% figure is the pre-registered target
from docs/validation_targets.md, not a number chosen once the test was seen to
pass. The NumPy prototype run before any solver code existed predicted 0.2-0.4%
at 25 steps per collision, so 2% carries roughly a 5x margin; if a measurement
lands between 0.4% and 2% that is a real regression worth explaining, even
though the assertion still passes.
"""

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.forces import NormalContactModel
from warp_dem.materials import (
    MaterialProperties,
    damping_ratio,
    effective_shear_modulus,
    wall_shear_modulus,
    wall_youngs_modulus,
)
from warp_dem.precision import cancellation_bound
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import (
    compute_budget,
    effective_youngs_modulus,
    hertz_static_overlap,
    shear_modulus,
)

GLASS = MaterialProperties(
    name="test_glass",
    density=2500.0,
    radius=0.002,
    youngs_modulus=1.0e8,
    poisson_ratio=0.20,
    restitution=0.90,
    friction_particle=0.25,
    friction_wall=0.35,
    rolling_friction=0.01,
    restitution_wall=0.90,
)

NO_GRAVITY = (0.0, 0.0, 0.0)


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _two_particles(device, separation, v_approach, material=GLASS):
    """Two spheres on the x axis, centres `separation` apart, closing at v_approach."""
    state = ParticleState.allocate(2, device, material.radius, material.density)
    state.set_positions([[-separation / 2, 0.0, 0.0], [separation / 2, 0.0, 0.0]])
    state.set_velocities([[v_approach / 2, 0.0, 0.0], [-v_approach / 2, 0.0, 0.0]])
    return state


def _analytic_hertz(delta, material):
    """F = (4/3) E* sqrt(R*) delta^1.5, computed independently of the kernel."""
    e_eff = effective_youngs_modulus(material.youngs_modulus, material.poisson_ratio)
    r_eff = material.radius / 2.0
    return (4.0 / 3.0) * e_eff * math.sqrt(r_eff) * delta**1.5


# ── material constants ────────────────────────────────────────────────────────


def test_damping_ratio_is_zero_for_a_perfectly_elastic_contact():
    assert damping_ratio(1.0) == 0.0


def test_damping_ratio_approaches_minus_one_as_restitution_vanishes():
    """Fully plastic is a finite limit, not a singularity — but it is approached
    LOGARITHMICALLY, which is slower than intuition suggests.

    beta = ln(e)/sqrt(ln^2(e) + pi^2) needs |ln(e)| >> pi to saturate, so
    beta < -0.99 requires e < 2.7e-10. Measured: e=1e-2 gives -0.826, e=1e-6
    gives only -0.975. The practical consequence is that no physically
    meaningful restitution gets anywhere near the -1 limit, so the damping
    coefficient never becomes stiff and the model has no low-e pathology.
    """
    assert damping_ratio(1e-2) == pytest.approx(-0.826085, abs=1e-5)
    assert damping_ratio(1e-6) == pytest.approx(-0.975107, abs=1e-5)
    assert damping_ratio(1e-14) == pytest.approx(-1.0, abs=1e-2)
    assert -1.0 < damping_ratio(0.01) < damping_ratio(0.5) < damping_ratio(0.99) < 0.0


def test_damping_ratio_rejects_unphysical_restitution():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="restitution"):
            damping_ratio(bad)


def test_wall_moduli_are_exactly_twice_the_pair_moduli():
    """A rigid wall contributes no compliance, so only one term survives."""
    e, nu = GLASS.youngs_modulus, GLASS.poisson_ratio
    assert wall_youngs_modulus(e, nu) == pytest.approx(2.0 * effective_youngs_modulus(e, nu))
    assert wall_shear_modulus(e, nu) == pytest.approx(2.0 * effective_shear_modulus(e, nu))


def test_effective_shear_modulus_matches_the_mindlin_definition():
    """1/G* = 2 (2 - nu) / G for identical spheres."""
    e, nu = GLASS.youngs_modulus, GLASS.poisson_ratio
    expected = 1.0 / (2.0 * (2.0 - nu) / shear_modulus(e, nu))
    assert effective_shear_modulus(e, nu) == pytest.approx(expected)


def test_material_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="density"):
        dataclasses.replace(GLASS, density=-1.0)
    with pytest.raises(ValueError, match="poisson_ratio"):
        dataclasses.replace(GLASS, poisson_ratio=0.6)
    with pytest.raises(ValueError, match="friction_particle"):
        dataclasses.replace(GLASS, friction_particle=-0.1)


# ── force magnitude ───────────────────────────────────────────────────────────


def test_no_force_when_particles_are_apart(device):
    state = _two_particles(device, separation=3.0 * GLASS.radius, v_approach=0.0)
    Solver(state, dt=1e-6, gravity=NO_GRAVITY,
           contact_model=NormalContactModel(state, GLASS.pair_params()))
    np.testing.assert_allclose(state.forces(), 0.0, atol=1e-12)


def test_no_force_when_particles_exactly_touch(device):
    """Zero overlap is zero force, so admitting the pair would cost a slot for nothing."""
    state = _two_particles(device, separation=2.0 * GLASS.radius, v_approach=0.0)
    Solver(state, dt=1e-6, gravity=NO_GRAVITY,
           contact_model=NormalContactModel(state, GLASS.pair_params()))
    np.testing.assert_allclose(state.forces(), 0.0, atol=1e-12)


@pytest.mark.parametrize("delta_ratio", [1e-4, 1e-3, 1e-2])
def test_static_force_matches_the_analytic_hertz_law(device, delta_ratio):
    """Zero relative velocity isolates the elastic term from the damping term.

    The tolerance is DERIVED from float32 cancellation, not chosen. Overlap is
    formed by subtracting two lengths of order 2R to leave one of order delta,
    so its relative precision is EPS * 2R / delta, and the delta^1.5 force law
    amplifies that by 1.5. At delta/R = 1e-4 that bound is 3.6e-3 — nearly four
    orders looser than at delta/R = 1e-2, for the same code. See
    precision.cancellation_bound and docs/precision.md.
    """
    delta = delta_ratio * GLASS.radius
    state = _two_particles(device, separation=2.0 * GLASS.radius - delta, v_approach=0.0)
    Solver(state, dt=1e-6, gravity=NO_GRAVITY,
           contact_model=NormalContactModel(state, GLASS.pair_params()))

    tol = cancellation_bound(2.0 * GLASS.radius, delta, exponent=1.5)
    forces = state.forces()
    expected = _analytic_hertz(delta, GLASS)
    # Particle 0 sits on the left, so repulsion pushes it further left.
    assert forces[0][0] == pytest.approx(-expected, rel=tol)
    assert forces[1][0] == pytest.approx(expected, rel=tol)
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1e-9)


def test_overlap_precision_degrades_as_the_overlap_shrinks(device):
    """Characterisation, not a pass/fail: the error scales as 1/delta.

    This is the signature of cancellation rather than of a bug, and pinning it
    down is what turned a failing tolerance in Block 10 into a documented
    property. A solver defect would not care how deep the contact is.
    """
    errors = {}
    for delta_ratio in (1e-4, 1e-3, 1e-2):
        delta = delta_ratio * GLASS.radius
        state = _two_particles(device, separation=2.0 * GLASS.radius - delta, v_approach=0.0)
        Solver(state, dt=1e-6, gravity=NO_GRAVITY,
               contact_model=NormalContactModel(state, GLASS.pair_params()))
        measured = abs(float(state.forces()[0][0]))
        expected = _analytic_hertz(delta, GLASS)
        errors[delta_ratio] = abs(measured - expected) / expected

    for ratio, err in errors.items():
        assert err <= cancellation_bound(2.0 * GLASS.radius, ratio * GLASS.radius, 1.5)
    # Ten times the overlap, roughly ten times the precision.
    assert errors[1e-4] > errors[1e-3] > errors[1e-2]


def test_force_scales_as_delta_to_the_three_halves(device):
    """The exponent is the physics. A linear spring would give a ratio of 4."""
    model_params = GLASS.pair_params()
    magnitudes = []
    for delta in (1e-3 * GLASS.radius, 4e-3 * GLASS.radius):
        state = _two_particles(device, separation=2.0 * GLASS.radius - delta, v_approach=0.0)
        Solver(state, dt=1e-6, gravity=NO_GRAVITY,
               contact_model=NormalContactModel(state, model_params))
        magnitudes.append(abs(float(state.forces()[0][0])))

    assert magnitudes[1] / magnitudes[0] == pytest.approx(4.0**1.5, rel=1e-3)


def test_newtons_third_law_holds_for_every_contact(device):
    """Total force on an isolated, gravity-free cluster must vanish exactly."""
    rng = np.random.default_rng(0)
    n = 40
    r = GLASS.radius
    positions = rng.uniform(-4 * r, 4 * r, size=(n, 3))
    velocities = rng.uniform(-0.5, 0.5, size=(n, 3))

    state = ParticleState.allocate(n, device, r, GLASS.density)
    state.set_positions(positions)
    state.set_velocities(velocities)
    model = NormalContactModel(state, GLASS.pair_params())
    Solver(state, dt=1e-6, gravity=NO_GRAVITY, contact_model=model)

    assert model.contacts.last_count > 0, "test is vacuous without contacts"
    net = state.forces().sum(axis=0)
    scale = np.abs(state.forces()).max()
    np.testing.assert_allclose(net, 0.0, atol=1e-5 * max(scale, 1.0))


def test_damping_opposes_relative_normal_motion(device):
    """Approach adds to the repulsion; separation subtracts from it."""
    delta = 1e-3 * GLASS.radius
    params = GLASS.pair_params()
    magnitudes = {}
    for label, v in (("approach", 1.0), ("static", 0.0), ("separate", -1.0)):
        state = _two_particles(device, separation=2.0 * GLASS.radius - delta, v_approach=v)
        Solver(state, dt=1e-6, gravity=NO_GRAVITY,
               contact_model=NormalContactModel(state, params))
        magnitudes[label] = -float(state.forces()[0][0])

    assert magnitudes["approach"] > magnitudes["static"] > magnitudes["separate"]
    # The elastic term is symmetric, so the damping contributions must be too.
    assert (magnitudes["approach"] - magnitudes["static"]) == pytest.approx(
        magnitudes["static"] - magnitudes["separate"], rel=1e-3
    )


def test_static_overlap_inverts_to_hertz_static_overlap(device):
    """The kernel and the host-side inverse must agree on the same contact."""
    delta = 5e-4 * GLASS.radius
    state = _two_particles(device, separation=2.0 * GLASS.radius - delta, v_approach=0.0)
    Solver(state, dt=1e-6, gravity=NO_GRAVITY,
           contact_model=NormalContactModel(state, GLASS.pair_params()))

    measured_force = abs(float(state.forces()[0][0]))
    recovered = hertz_static_overlap(
        measured_force, GLASS.radius, GLASS.youngs_modulus, GLASS.poisson_ratio
    )
    assert recovered == pytest.approx(delta, rel=1e-3)


# ── restitution: Validation Target 4a ─────────────────────────────────────────


def measure_restitution(device, restitution, v_approach=1.0, steps_per_collision=25):
    """Run one head-on collision and return the measured coefficient of restitution.

    Starts the particles just touching so the whole run is the collision, and
    stops once they are apart again. Polling for separation costs a host sync
    per chunk, which is exactly the kind of thing that must never appear in a
    production loop and is entirely fine in a test.
    """
    material = dataclasses.replace(GLASS, restitution=restitution)
    budget = compute_budget(
        material.radius, material.density, material.youngs_modulus,
        material.poisson_ratio, impact_velocity=v_approach,
        hertz_steps=steps_per_collision,
    )
    # min(Rayleigh, Hertz), never the Hertz bound alone — see scripts/restitution.py.
    dt = budget.limit

    state = _two_particles(device, separation=2.0 * material.radius, v_approach=v_approach)
    solver = Solver(state, dt=dt, gravity=NO_GRAVITY, budget=budget,
                    contact_model=NormalContactModel(state, material.pair_params()))

    touched = False
    for _ in range(400):
        solver.run(5)
        gap = float(state.positions()[1][0] - state.positions()[0][0])
        overlapping = gap < 2.0 * material.radius
        if overlapping:
            touched = True
        elif touched:
            break
    else:
        raise AssertionError("collision never completed")

    assert touched, "particles never made contact — the test would be vacuous"
    v = state.velocities()
    return abs(float(v[0][0] - v[1][0])) / v_approach


@pytest.mark.parametrize("restitution", [0.5, 0.7, 0.9])
def test_restitution_matches_configured_value(device, restitution):
    """VALIDATION TARGET 4a: measured e within 2% of the configured e."""
    measured = measure_restitution(device, restitution)
    relative_error = abs(measured - restitution) / restitution
    assert relative_error < 0.02, (
        f"configured e={restitution}, measured e={measured:.5f}, "
        f"error {100 * relative_error:.3f}% exceeds the 2% target"
    )


def test_perfectly_elastic_collision_loses_no_speed(device):
    """beta = 0 must mean exactly no dissipation, not merely very little."""
    measured = measure_restitution(device, 1.0)
    assert measured == pytest.approx(1.0, abs=5e-3)
