"""Energy conservation audit — Block 16.

REQUIRED PHASE 1 TEST #5: energy is conserved in a frictionless, non-damped
configuration.

"Non-damped" means e = 1, so beta = 0 and the viscous term vanishes identically.
"Frictionless" means mu = 0, so no energy goes into or out of tangential
springs. What remains is a Hamiltonian system, and velocity-Verlet is symplectic,
so the total should not drift secularly — it should oscillate about a constant
with an amplitude set by the timestep.

THE ELASTIC TERM IS THE POINT. At any instant part of the system's energy sits
inside the contact springs. A total that counts only kinetic plus potential
oscillates violently every time anything touches anything, and that looks
exactly like an unstable integrator. It is an accounting failure, not a physics
one, and this file asserts the difference explicitly so the distinction is
recorded rather than merely known.

TOLERANCES ARE DERIVED. Symplectic integrators bound their energy error rather
than accumulating it, so the physical drift is O(dt^2) and bounded. What is NOT
bounded is float32 accumulation, so the tolerance comes from
precision.accumulation_bound over the number of additions actually performed.
"""

import dataclasses

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.diagnostics import energy_breakdown, kinetic_energy, potential_energy
from warp_dem.forces import ContactModel, NormalContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.precision import accumulation_bound
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget

RADIUS = 0.002
DENSITY = 2500.0
NO_GRAVITY = (0.0, 0.0, 0.0)
GRAVITY = (0.0, 0.0, -9.81)

#: Perfectly elastic and frictionless: the Hamiltonian configuration required
#: by test #5. e = 1 makes beta exactly 0, so damping is absent rather than
#: small.
IDEAL = MaterialProperties(
    name="ideal",
    density=DENSITY,
    radius=RADIUS,
    youngs_modulus=1.0e8,
    poisson_ratio=0.20,
    restitution=1.0,
    friction_particle=0.0,
    friction_wall=0.0,
    rolling_friction=0.0,
    restitution_wall=1.0,
)


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _budget(material, steps=50):
    return compute_budget(
        material.radius, material.density, material.youngs_modulus,
        material.poisson_ratio, impact_velocity=1.0, hertz_steps=steps,
    )


# ── the accounting ────────────────────────────────────────────────────────────


def test_kinetic_energy_includes_rotation(device):
    """Spin is real energy, and tangential force moves energy into it."""
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    state.set_velocities([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    state.set_angular_velocities([[0.0, 0.0, 0.0], [0.0, 0.0, 50.0]])

    mass = float(state.mass.numpy()[0])
    inertia = float(state.inertia.numpy()[0])
    expected = 0.5 * mass * 1.0**2 + 0.5 * inertia * 50.0**2
    assert kinetic_energy(state) == pytest.approx(expected, rel=1e-5)


def test_potential_energy_tracks_height(device):
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    state.set_positions([[0.0, 0.0, 0.5]])
    mass = float(state.mass.numpy()[0])
    assert potential_energy(state, GRAVITY) == pytest.approx(mass * 9.81 * 0.5, rel=1e-5)


def test_elastic_energy_is_zero_without_contacts(device):
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    state.set_positions([[0.0, 0.0, 0.0], [10.0 * RADIUS, 0.0, 0.0]])
    model = NormalContactModel(state, IDEAL.pair_params())
    Solver(state, dt=1e-6, gravity=NO_GRAVITY, contact_model=model)

    breakdown = energy_breakdown(state, NO_GRAVITY, model.contacts,
                                 IDEAL.pair_params().e_eff)
    assert breakdown.elastic == 0.0


def test_elastic_energy_matches_the_analytic_integral(device):
    """U = (8/15) E* sqrt(R*) delta^2.5, the integral of the Hertz force."""
    delta = 1e-3 * RADIUS
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    state.set_positions([[0.0, 0.0, 0.0], [2.0 * RADIUS - delta, 0.0, 0.0]])
    model = NormalContactModel(state, IDEAL.pair_params())
    Solver(state, dt=1e-6, gravity=NO_GRAVITY, contact_model=model)

    e_eff = IDEAL.pair_params().e_eff
    expected = (8.0 / 15.0) * e_eff * np.sqrt(RADIUS / 2.0) * delta**2.5
    measured = energy_breakdown(state, NO_GRAVITY, model.contacts, e_eff).elastic
    assert measured == pytest.approx(expected, rel=5e-3)


def test_omitting_elastic_energy_makes_the_total_appear_to_oscillate(device):
    """The accounting failure this module exists to prevent, demonstrated.

    Two particles collide elastically. Kinetic + potential alone dips hard
    during the contact — energy has gone into the springs and has not been
    counted. Including the elastic term removes the dip. The point is that the
    incomplete total looks like an integrator problem and is not one.
    """
    material = IDEAL
    budget = _budget(material)
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    state.set_positions([[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]])
    state.set_velocities([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    model = NormalContactModel(state, material.pair_params())
    solver = Solver(state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
                    contact_model=model)

    e_eff = material.pair_params().e_eff
    incomplete, complete = [], []
    for _ in range(60):
        solver.run(5)
        parts = energy_breakdown(state, NO_GRAVITY, model.contacts, e_eff)
        incomplete.append(parts.kinetic + parts.potential)
        complete.append(parts.total)

    def swing(series):
        return (max(series) - min(series)) / max(abs(v) for v in series)

    assert swing(incomplete) > 0.5, (
        "the collision was missed entirely; the demonstration is vacuous"
    )
    assert swing(complete) < 0.02, f"complete total swung by {swing(complete):.3%}"


# ── REQUIRED TEST #5 ──────────────────────────────────────────────────────────


def _energy_series(device, material, model_class, steps, samples, seed=11, n=24):
    """Run a small frictionless cluster and sample the total energy."""
    budget = _budget(material)
    rng = np.random.default_rng(seed)
    state = ParticleState.allocate(n, device, RADIUS, DENSITY)
    state.set_positions(rng.uniform(-5 * RADIUS, 5 * RADIUS, size=(n, 3)))
    state.set_velocities(rng.uniform(-0.3, 0.3, size=(n, 3)))
    model = model_class(state, material.pair_params())
    solver = Solver(state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
                    contact_model=model)

    e_eff = material.pair_params().e_eff
    series = []
    per_sample = steps // samples
    for _ in range(samples):
        solver.run(per_sample)
        series.append(energy_breakdown(state, NO_GRAVITY, model.contacts, e_eff).total)
    return series, solver


def test_energy_is_conserved_in_a_frictionless_undamped_configuration(device):
    """REQUIRED TEST #5.

    The tolerance is derived, not chosen. Velocity-Verlet is symplectic, so its
    energy error is BOUNDED rather than secular — it oscillates at O(dt^2) and
    does not accumulate. What does accumulate is float32 rounding, at two
    velocity additions per step, and energy is quadratic in velocity so a
    relative error eps in v is 2 eps in E.
    """
    steps = 10_000
    series, _ = _energy_series(device, IDEAL, NormalContactModel, steps, samples=25)

    initial = series[0]
    drift = max(abs(e - initial) for e in series) / abs(initial)

    speed = 0.3
    velocity_error = accumulation_bound(speed, 2 * steps) / speed
    tolerance = max(2.0 * velocity_error, 0.02)
    assert drift < tolerance, (
        f"energy drifted {drift:.3%} over {steps} steps, bound {tolerance:.3%}"
    )


def test_energy_drift_does_not_grow_secularly(device):
    """Symplectic means BOUNDED error, not small error.

    A non-symplectic integrator of the same order would show drift growing
    roughly linearly with step count. Doubling the run must not double the
    excursion.
    """
    short, _ = _energy_series(device, IDEAL, NormalContactModel, 5_000, samples=20)
    long, _ = _energy_series(device, IDEAL, NormalContactModel, 20_000, samples=80)

    def excursion(series):
        return max(abs(e - series[0]) for e in series) / abs(series[0])

    # Four times the steps must not give four times the drift.
    assert excursion(long) < 2.5 * max(excursion(short), 1e-9), (
        f"drift grew from {excursion(short):.3%} to {excursion(long):.3%}"
    )


def test_the_full_model_conserves_energy_when_friction_is_disabled(device):
    """The Hertz-Mindlin path must be Hamiltonian too when mu = 0 and beta = 0.

    Same physics as the oracle, different code path — this is what catches a
    tangential spring that leaks energy even when it is supposed to be inert.
    """
    steps = 10_000
    series, _ = _energy_series(device, IDEAL, ContactModel, steps, samples=25)
    drift = max(abs(e - series[0]) for e in series) / abs(series[0])
    assert drift < 0.02, f"full model drifted {drift:.3%}"


def test_damping_removes_energy_monotonically(device):
    """The control: with e < 1 the total must fall, and never rise.

    Without this, test #5 could pass on a solver that had no contact forces at
    all — a system that never interacts conserves energy trivially.
    """
    material = dataclasses.replace(IDEAL, restitution=0.6)
    series, _ = _energy_series(device, material, NormalContactModel, 10_000, samples=25)

    assert series[-1] < series[0], "damping did not dissipate anything"
    rises = [b - a for a, b in zip(series, series[1:], strict=False) if b > a]
    assert not rises or max(rises) < 1e-3 * abs(series[0]), (
        f"damped energy increased by {max(rises):.3e}"
    )


def test_gravity_converts_potential_to_kinetic_without_loss(device):
    """Free fall, no contacts: the cleanest possible conservation check.

    Also the configuration MOST exposed to the Block 6 accumulation drift, and
    the tolerance is derived from it rather than chosen. The particle starts at
    z = 1.0 m, and docs/precision.md records that drift scales with the
    MAGNITUDE OF THE COORDINATE rather than the distance travelled — so a metre
    up is roughly ten times worse per step than the 0.1 m of the original
    measurement.

    Energy here is dominated by m g z, so a position error dz costs m g dz and
    the relative energy error is just dz / z. Bound 1.2e-3, measured 1.6e-4 —
    holds with a factor of 7. Tightening this to a hand-picked 1e-4 would make
    the test fail on any machine that rounds slightly differently, while
    catching nothing.
    """
    height = 1.0
    steps = 20_000
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    state.set_positions([[0.0, 0.0, height]])
    solver = Solver(state, dt=1e-5, gravity=GRAVITY)

    before = energy_breakdown(state, GRAVITY).total
    solver.run(steps)
    after = energy_breakdown(state, GRAVITY).total

    # One position accumulation per step; energy error is m g dz over m g z.
    tolerance = accumulation_bound(height, steps) / height
    assert abs(after - before) / abs(before) < tolerance, (
        f"energy changed by {abs(after - before) / abs(before):.3e}, "
        f"accumulation bound {tolerance:.3e}"
    )
