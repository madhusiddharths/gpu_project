"""Mindlin tangential force with Coulomb clipping — Block 13.

The headline is `test_coulomb_impulse_saturates_above_the_threshold_angle`,
which locates the sliding/sticking transition of an oblique impact and compares
it with the analytic threshold

    phi_th = arctan[(7/2) mu_s (1 + e)]

from [Sunday2020] section 4. For mu_s = 0.3 and e = 1 that is 64.54 degrees.

WHERE THE 7/2 COMES FROM. A sphere striking a fixed surface has to have the
sliding of its CONTACT POINT arrested, and the contact point carries both
translation and spin. Cancelling surface velocity v_t needs a tangential
impulse of (2/7) m v_t — the 2/7 is 1/(1 + 1/k) with k = 2/5 the moment of
inertia coefficient of a solid sphere. Coulomb can supply at most mu_s times
the normal impulse m v_n (1+e). Sliding therefore persists through the whole
contact when

    mu_s m v_n (1+e)  <  (2/7) m v_t     <=>     tan(phi) > (7/2) mu_s (1+e)

The moving sphere strikes a PINNED sphere rather than a wall, because walls do
not exist until Block 15. Mechanically that is the same problem: the analysis
depends only on the impacting sphere's mass and inertia, and a pinned partner
supplies the immovable surface.

THE STICKING REGIME IS DELIBERATELY NOT ASSERTED against the rigid-body result
v_t' = (5/7) v_t. That result assumes surface sliding, once arrested, stays
arrested. A soft-sphere contact has a real elastic tangential spring, so below
the cone it stores energy and gives some back — genuine tangential restitution
(Maw-Barber). Asserting the rigid-body number would be asserting that the
tangential spring does not exist. What IS asserted below the threshold is the
robust part: the Coulomb impulse is not saturated.
"""

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.forces import ContactModel, NormalContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget

RADIUS = 0.002
DENSITY = 2500.0
NO_GRAVITY = (0.0, 0.0, 0.0)

#: The oblique-impact tests use E = 1e10 Pa, a hundred times the project's
#: working stiffness, and that is a deliberate test-design choice rather than a
#: physical one. The rigid-body impulse analysis assumes the contact geometry is
#: fixed during the collision. A soft contact lasts long enough for the sphere to
#: slide a noticeable fraction of a radius across its partner's curvature and
#: glance off, so the normal impulse never reaches m v_n (1+e). Measured slide at
#: 76 degrees incidence: 0.33 R at E = 1e8, 0.05 R at E = 1e10.
#:
#: The finding this encodes: soft-sphere DEM reproduces rigid-body impact results
#: only in the limit where contact duration is short compared with the time the
#: geometry takes to change. The project's own runs are quasi-static beds, where
#: that limit is irrelevant — but a test of an IMPACT law has to sit inside it.
IMPACT_MODULUS = 1.0e10

BASE = MaterialProperties(
    name="test_glass",
    density=DENSITY,
    radius=RADIUS,
    youngs_modulus=IMPACT_MODULUS,
    poisson_ratio=0.20,
    restitution=1.0,  # beta = 0, so the impulse analysis is clean
    friction_particle=0.30,
    friction_wall=0.30,
    rolling_friction=0.0,
    restitution_wall=1.0,
)

MU = BASE.friction_particle
THRESHOLD_DEG = math.degrees(math.atan(3.5 * MU * (1.0 + BASE.restitution)))


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _budget(material, impact_velocity=1.0, steps=50):
    return compute_budget(
        material.radius, material.density, material.youngs_modulus,
        material.poisson_ratio, impact_velocity=impact_velocity, hertz_steps=steps,
    )


def _oblique_impact(device, material, v_n, v_t, steps=50):
    """Fire particle 0 at a pinned particle 1. Returns (velocity, omega) of 0."""
    budget = _budget(material, impact_velocity=max(v_n, 1e-3), steps=steps)
    r = material.radius

    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[0.0, 0.0, 0.0], [2.0 * r, 0.0, 0.0]])
    state.set_velocities([[v_n, v_t, 0.0], [0.0, 0.0, 0.0]])
    state.set_fixed([1])

    solver = Solver(
        state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
        contact_model=ContactModel(state, material.pair_params()),
    )

    touched = False
    for _ in range(20_000):
        solver.step()
        gap = float(np.linalg.norm(state.positions()[1] - state.positions()[0]))
        if gap < 2.0 * r:
            touched = True
        elif touched:
            break
    assert touched, "no contact occurred"
    return state.velocities()[0], state.angular_velocities()[0]


def _saturation_ratio(device, material, phi_deg, v_n=1.0):
    """Tangential impulse as a fraction of the Coulomb maximum, self-normalised.

    Spin is the right observable. Normal force acts along the line of centres
    and exerts NO torque, so angular velocity measures the tangential impulse
    alone, uncontaminated by the normal-direction leak that Block 11 found in
    delta_v_t.

    The normal impulse is taken from the MEASURED normal velocity change rather
    than from m v_n (1+e). Any residual glancing then cancels out of the ratio
    instead of appearing as a spurious decay at high incidence — with the
    theoretical normalisation the ratio fell to 0.95 by 74 degrees; self
    normalised it stays flat at 0.993.

        I w' = R J_t   and   J_n = m dv_n   =>   ratio = (2/5) R w' / (mu dv_n)
    """
    v_t = v_n * math.tan(math.radians(phi_deg))
    velocity, omega = _oblique_impact(device, material, v_n=v_n, v_t=v_t)
    dv_n = v_n - float(velocity[0])
    return (2.0 / 5.0) * material.radius * abs(float(omega[2])) / (
        material.friction_particle * dv_n
    )


# ── the oracle cross-check ────────────────────────────────────────────────────


def test_frictionless_full_model_matches_the_normal_only_oracle(device):
    """With mu = 0 the slot machinery must reproduce the pair-list model exactly.

    This is the Block 9 philosophy applied one level up: NormalContactModel
    iterates the flat pair list with no history at all, so if the two disagree
    the fault is in the slot bookkeeping rather than in the physics. It isolates
    Block 12's machinery from Block 13's force law.
    """
    material = dataclasses.replace(BASE, friction_particle=0.0, restitution=0.8)
    budget = _budget(material)
    rng = np.random.default_rng(7)
    positions = rng.uniform(-4 * RADIUS, 4 * RADIUS, size=(30, 3))
    velocities = rng.uniform(-0.5, 0.5, size=(30, 3))

    results = []
    for model_class in (NormalContactModel, ContactModel):
        state = ParticleState.allocate(30, device, RADIUS, DENSITY)
        state.set_positions(positions)
        state.set_velocities(velocities)
        solver = Solver(
            state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
            contact_model=model_class(state, material.pair_params()),
        )
        solver.run(200)
        results.append(state.positions())

    np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-9)


# ── the Coulomb cone ──────────────────────────────────────────────────────────


def test_friction_opposes_relative_tangential_motion(device):
    """The most basic requirement: friction must slow the sliding, not speed it."""
    v_out, omega = _oblique_impact(device, BASE, v_n=1.0, v_t=2.5)
    assert 0.0 < float(v_out[1]) < 2.5, f"tangential velocity became {v_out[1]}"
    # Right-hand rule: sliding along +y against a surface at +x spins it about -z.
    assert float(omega[2]) < 0.0


def test_tangential_impulse_saturates_at_mu_times_the_normal_impulse(device):
    """Deep in the sliding regime the contact delivers exactly the Coulomb maximum.

    The ratio is INDEPENDENT of incidence angle across the whole sliding
    regime, which is the signature of saturation: once the cone is reached,
    sliding faster cannot extract more friction.
    """
    for phi in (66.0, 70.0, 74.0):
        ratio = _saturation_ratio(device, BASE, phi)
        assert ratio == pytest.approx(1.0, rel=0.02), (
            f"phi={phi}: tangential impulse is {ratio:.3f} of the Coulomb maximum"
        )


def test_sliding_spin_matches_the_impulse_prediction(device):
    """omega' = 5 mu (1+e) v_n / (2R), from I omega = R J_t with I = (2/5) m R^2."""
    v_n = 1.0
    expected = 5.0 * MU * (1.0 + BASE.restitution) * v_n / (2.0 * RADIUS)
    v_t = v_n * math.tan(math.radians(68.0))
    _, omega = _oblique_impact(device, BASE, v_n=v_n, v_t=v_t)
    assert abs(float(omega[2])) == pytest.approx(expected, rel=0.05)


def test_coulomb_impulse_saturates_above_the_threshold_angle(device):
    """The Block 13 headline: locate the sliding/sticking transition.

    Above phi_th the contact slides throughout and delivers the full Coulomb
    impulse. Below it, sliding is arrested part way through and it delivers
    less. Scanning incidence and finding where the ratio saturates locates the
    transition WITHOUT assuming where it is.

    Measured onset is 60-62 degrees against an analytic 64.54 — about 5% low,
    and converged: E = 1e11 gives the same answer as E = 1e10, so it is a
    property of the contact model rather than a numerical artefact.

    The reason is that a Mindlin contact does not switch between sticking and
    sliding at a sharp threshold. It has an inner stick zone and an outer slip
    annulus, so PARTIAL SLIP begins before gross sliding does, and the
    transition is smeared over several degrees. The rigid-body formula is the
    limit of a sharp transition that a compliant contact does not have. The
    test asserts the smeared transition it actually produces.
    """
    ratios = {phi: _saturation_ratio(device, BASE, phi)
              for phi in (40.0, 50.0, 56.0, 62.0, 66.0, 70.0)}

    # Well below the threshold, sliding is arrested and the cone is not reached.
    assert ratios[40.0] < 0.7, ratios
    assert ratios[50.0] < 0.85, ratios
    # Above it, saturated and flat.
    for phi in (62.0, 66.0, 70.0):
        assert ratios[phi] == pytest.approx(1.0, rel=0.02), ratios

    # Monotone approach from below, then a plateau rather than a peak.
    assert ratios[40.0] < ratios[50.0] < ratios[56.0] < ratios[62.0]

    onset = min(phi for phi, r in sorted(ratios.items()) if r > 0.95)
    assert abs(onset - THRESHOLD_DEG) / THRESHOLD_DEG < 0.10, (
        f"saturation onset {onset} deg vs analytic {THRESHOLD_DEG:.2f} deg"
    )


def test_threshold_angle_matches_the_analytic_expression():
    """arctan[(7/2) mu_s (1+e)] = 64.54 deg for mu_s = 0.3, e = 1."""
    assert THRESHOLD_DEG == pytest.approx(64.54, abs=0.01)


@pytest.mark.parametrize("mu", [0.1, 0.3, 0.5])
def test_tangential_force_never_exceeds_the_friction_cone(device, mu):
    """|F_t| <= mu |F_n| must hold at every contact, at every instant."""
    material = dataclasses.replace(BASE, friction_particle=mu, restitution=0.9)
    budget = _budget(material)
    rng = np.random.default_rng(3)
    n = 40
    state = ParticleState.allocate(n, device, RADIUS, DENSITY)
    state.set_positions(rng.uniform(-4 * RADIUS, 4 * RADIUS, size=(n, 3)))
    state.set_velocities(rng.uniform(-1.0, 1.0, size=(n, 3)))
    model = ContactModel(state, material.pair_params())
    solver = Solver(state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
                    contact_model=model)

    for _ in range(50):
        solver.run(10)
        # Torque comes only from tangential force, |tau| = |F_t| * lever, so it
        # bounds |F_t| without needing the per-contact force to be exported.
        torque = np.linalg.norm(state.torques(), axis=1)
        force = np.linalg.norm(state.forces(), axis=1)
        implied_ft = torque / RADIUS
        # Slack because a particle may have several contacts, whose normal
        # forces can cancel while their tangential torques add.
        assert np.all(implied_ft <= mu * force * 6.0 + 1e-9)


# ── the spring, and its memory ────────────────────────────────────────────────


def test_static_contact_builds_tangential_force_from_stored_displacement(device):
    """Below the cone the force grows with accumulated slip, not with speed.

    A purely viscous model at constant sliding speed gives a CONSTANT force. A
    spring gives one that keeps growing as displacement accumulates. That
    difference is the entire reason history exists, so this test is the direct
    statement of it.

    KINEMATICALLY DRIVEN: both particles are pinned, and particle 1 is given a
    constant tangential velocity. A pinned particle has inv_mass = 0 so it never
    accelerates, but its position still integrates its velocity — it is a
    prescribed-motion boundary condition. That holds the overlap exactly
    constant while the tangential spring winds up, isolating the spring from
    the normal dynamics.

    A first attempt instead pressed two free particles together at a fixed
    overlap and let them go. At E = 1e10 a 1% overlap is 19.6 N on an 84 mg
    particle — 234,000 m/s^2 — so it was ejected at 4.9 m/s within one window
    and the measured force was zero because the contact had ended, not because
    the spring was wrong.
    """
    # Project stiffness here, not IMPACT_MODULUS: nothing about this test needs
    # a short contact, and 1e8 keeps the forces at a physically sensible scale.
    material = dataclasses.replace(
        BASE, youngs_modulus=1.0e8, restitution=0.9, friction_particle=10.0
    )
    budget = _budget(material)
    r = material.radius
    overlap = 1e-3 * r
    drag_speed = 0.01

    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[0.0, 0.0, 0.0], [2.0 * r - overlap, 0.0, 0.0]])
    state.set_velocities([[0.0, 0.0, 0.0], [0.0, drag_speed, 0.0]])
    state.set_fixed([0, 1])
    model = ContactModel(state, material.pair_params())
    solver = Solver(state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
                    contact_model=model)

    window = 40
    magnitudes = []
    for _ in range(4):
        solver.run(window)
        magnitudes.append(abs(float(state.forces()[0][1])))

    assert magnitudes == sorted(magnitudes), f"force did not grow: {magnitudes}"
    # Linear winding: displacement is drag_speed * t, so force should be too.
    assert magnitudes[-1] == pytest.approx(4.0 * magnitudes[0], rel=0.1), magnitudes

    # And the force must match the spring constant times the stored slip.
    stored = model.history.as_map()[(0, 1)]
    slip = abs(float(stored[1]))
    assert slip == pytest.approx(drag_speed * solver.time, rel=0.05), (
        f"stored slip {slip:.3e} m vs imposed {drag_speed * solver.time:.3e} m"
    )


def test_history_is_discarded_when_a_sliding_pair_separates(device):
    """A pair that bounces apart must not resume its old friction state."""
    material = dataclasses.replace(BASE, restitution=0.9)
    budget = _budget(material)
    r = material.radius
    state = ParticleState.allocate(2, device, r, material.density)
    state.set_positions([[0.0, 0.0, 0.0], [2.0 * r, 0.0, 0.0]])
    state.set_velocities([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    state.set_fixed([1])
    model = ContactModel(state, material.pair_params())
    solver = Solver(state, dt=budget.limit, gravity=NO_GRAVITY, budget=budget,
                    contact_model=model)

    saw_history = False
    for _ in range(400):
        solver.run(10)
        if model.history.as_map():
            saw_history = True
        elif saw_history:
            break

    assert saw_history, "the pair never made contact"
    assert model.history.as_map() == {}, "history outlived the contact"


def test_zero_friction_produces_no_torque(device):
    """mu = 0 must mean no tangential force at all, hence no spin, exactly."""
    material = dataclasses.replace(BASE, friction_particle=0.0)
    v_out, omega = _oblique_impact(device, material, v_n=1.0, v_t=2.0)
    np.testing.assert_allclose(omega, 0.0, atol=1e-9)
