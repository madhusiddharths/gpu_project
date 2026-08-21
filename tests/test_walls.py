"""Axis-aligned box boundaries — Block 15.

Carries two of the five required Phase 1 unit tests:

  #3  a particle dropped on a plane reaches static equilibrium and stays there,
      with no jitter and no sinking
  #4  a particle on an inclined plane slides above the friction angle and holds
      below it

THE INCLINE IS BUILT BY TILTING GRAVITY, not by tilting a wall. A floor stays
axis-aligned at z = 0 and gravity becomes g(sin th, 0, -cos th). The two are the
same problem in a rotated frame, and it means the axis-aligned box is sufficient
for the whole of Phase 1 — a general oriented plane is not needed until the drum
in Phase 5.

A SPHERE ON A SLOPE DOES NOT SIMPLY "HOLD". The textbook block either slides or
stays put at tan th = mu. A sphere below the friction angle does not slide, but
it ROLLS, and a rolling sphere accelerates down the slope at (5/7) g sin th
regardless of how small the angle is — rolling has no threshold. What arrests it
is rolling resistance, which is precisely the physical content of Block 14's
correction term. So the honest form of test #4 is three claims, and this file
makes all three:

  a) above the friction angle the CONTACT POINT SLIPS
  b) below it the contact point does not slip — the sphere rolls instead
  c) with rolling resistance present, a shallow enough slope holds it completely
"""

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.materials import MaterialProperties
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget, hertz_static_overlap
from warp_dem.walls import WALL_COUNT, BoxBoundary

RADIUS = 0.002
DENSITY = 2500.0
G = 9.81

BASE = MaterialProperties(
    name="test_glass",
    density=DENSITY,
    radius=RADIUS,
    youngs_modulus=1.0e8,
    poisson_ratio=0.20,
    restitution=0.50,
    friction_particle=0.30,
    friction_wall=0.50,
    rolling_friction=0.0,
    restitution_wall=0.50,
)

BOUNDS_MIN = (-0.05, -0.05, 0.0)
BOUNDS_MAX = (0.05, 0.05, 0.05)


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


def _on_floor(device, material, gravity, height=None, velocity=(0.0, 0.0, 0.0),
              lock_rotation=False, bounds=None):
    """One particle resting on (or just above) the floor at z = 0."""
    r = material.radius
    state = ParticleState.allocate(1, device, r, material.density)
    state.set_positions([[0.0, 0.0, r if height is None else height]])
    state.set_velocities([list(velocity)])
    if lock_rotation:
        state.set_rotation_locked([0])

    lo, hi = bounds if bounds is not None else (BOUNDS_MIN, BOUNDS_MAX)
    budget = _budget(material)
    boundary = BoxBoundary(state, material.wall_params(), lo, hi)
    solver = Solver(state, dt=budget.limit, gravity=gravity, budget=budget,
                    boundary=boundary)
    return state, solver, boundary


# ── geometry and construction ─────────────────────────────────────────────────


def test_bounds_are_validated(device):
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    params = BASE.wall_params()
    with pytest.raises(ValueError, match="must exceed"):
        BoxBoundary(state, params, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="3-vectors"):
        BoxBoundary(state, params, (0.0, 0.0), (1.0, 1.0, 1.0))


def test_no_force_in_the_middle_of_the_box(device):
    state, _, _ = _on_floor(device, BASE, (0.0, 0.0, 0.0), height=0.025)
    np.testing.assert_allclose(state.forces(), 0.0, atol=1e-12)


def test_each_of_the_six_walls_repels(device):
    """Every wall must push INTO the domain, and only along its own axis."""
    material = dataclasses.replace(BASE, friction_wall=0.0)
    params = material.wall_params()
    lo = np.array(BOUNDS_MIN)
    hi = np.array(BOUNDS_MAX)
    overlap = 1e-3 * RADIUS

    for wall in range(WALL_COUNT):
        axis = wall % 3
        state = ParticleState.allocate(1, device, RADIUS, DENSITY)
        centre = (lo + hi) / 2.0
        if wall < 3:
            centre[axis] = lo[axis] + RADIUS - overlap
            expected_sign = 1.0
        else:
            centre[axis] = hi[axis] - RADIUS + overlap
            expected_sign = -1.0
        state.set_positions([centre.tolist()])
        boundary = BoxBoundary(state, params, BOUNDS_MIN, BOUNDS_MAX)
        Solver(state, dt=1e-6, gravity=(0.0, 0.0, 0.0), boundary=boundary)

        force = state.forces()[0]
        assert np.sign(force[axis]) == expected_sign, f"wall {wall}: {force}"
        others = [a for a in range(3) if a != axis]
        np.testing.assert_allclose(force[others], 0.0, atol=1e-12)


def test_wall_is_twice_as_stiff_as_a_particle_pair(device):
    """E*_wall = 2 E*_pair, so the same overlap gives twice the force."""
    material = dataclasses.replace(BASE, friction_wall=0.0)
    overlap = 1e-3 * RADIUS
    state, _, _ = _on_floor(device, material, (0.0, 0.0, 0.0),
                            height=RADIUS - overlap)
    measured = float(state.forces()[0][2])

    e_wall = material.wall_params().e_eff
    expected = (4.0 / 3.0) * e_wall * math.sqrt(RADIUS) * overlap**1.5
    assert measured == pytest.approx(expected, rel=2e-3)


# ── required test #3: rest on a plane, no jitter, no sinking ──────────────────


def _settle_on_floor(device, material, steps=60_000):
    state, solver, boundary = _on_floor(
        device, material, (0.0, 0.0, -G), height=1.5 * material.radius
    )
    solver.run(steps)
    return state, solver, boundary


def test_particle_dropped_on_a_plane_comes_to_rest(device):
    """REQUIRED TEST #3, part one: it stops."""
    state, _, _ = _settle_on_floor(device, BASE)
    speed = float(np.linalg.norm(state.velocities()[0]))
    assert speed < 1e-4, f"still moving at {speed:.3e} m/s"


def test_particle_at_rest_does_not_jitter(device):
    """REQUIRED TEST #3, part two: and it STAYS stopped.

    Jitter is the classic soft-contact failure: the particle sinks, is thrown
    out, falls back, and buzzes forever. It shows up as a position that keeps
    changing after the velocity has nominally settled, so the test watches the
    position over a long window rather than sampling it once.
    """
    state, solver, _ = _settle_on_floor(device, BASE)
    heights = []
    for _ in range(20):
        solver.run(500)
        heights.append(float(state.positions()[0][2]))

    spread = max(heights) - min(heights)
    assert spread < 1e-7, f"height varied by {spread:.3e} m: {heights[:5]}"


def test_particle_does_not_sink_into_the_floor(device):
    """REQUIRED TEST #3, part three: it rests at the Hertz equilibrium.

    The resting height is not the radius — it is the radius minus the static
    overlap at which Hertz repulsion balances the particle's own weight. That
    overlap is a PREDICTION, so this checks the number rather than merely
    asserting the particle is somewhere near the floor.
    """
    state, _, _ = _settle_on_floor(device, BASE)
    height = float(state.positions()[0][2])
    mass = float(state.mass.numpy()[0])

    # Sphere on a rigid plane: R* = r and E* is the wall value, so the
    # two-sphere helper is fed the doubled modulus and the full radius.
    e_wall = BASE.wall_params().e_eff
    expected_overlap = (mass * G / ((4.0 / 3.0) * e_wall * math.sqrt(RADIUS))) ** (
        2.0 / 3.0
    )
    assert height == pytest.approx(RADIUS - expected_overlap, rel=1e-4)
    assert expected_overlap / RADIUS < 0.01, "violates Validation Target 5"
    assert height > 0.0, "particle sank through the floor"


def test_resting_overlap_agrees_with_the_timestep_helper(device):
    """Cross-check against hertz_static_overlap, which uses the PAIR modulus.

    The helper assumes two identical spheres, so feeding it a wall contact means
    knowing the wall is twice as stiff and has R* = r rather than r/2. A sphere
    pair of radius 2r made of material 2E has exactly the wall's E* and R*.

    THE TOLERANCE IS IN ULPS, NOT PERCENT, and that is the point. The resting
    overlap here is 2.60e-7 m against a radius of 2e-3 m — 0.013% of R — so it
    spans only about 1100 float32 quanta of the coordinate. Everything about the
    equilibrium is coarse at that scale. See test_static_overlap_is_quantised.
    """
    state, _, _ = _settle_on_floor(device, BASE)
    mass = float(state.mass.numpy()[0])
    overlap = RADIUS - float(state.positions()[0][2])

    equivalent = hertz_static_overlap(
        mass * G, 2.0 * RADIUS, 2.0 * BASE.youngs_modulus, BASE.poisson_ratio
    )
    quantum = float(np.spacing(np.float32(RADIUS)))
    assert abs(overlap - equivalent) < 8.0 * quantum, (
        f"overlap {overlap:.6e} vs {equivalent:.6e}, a gap of "
        f"{abs(overlap - equivalent) / quantum:.1f} ulps of the coordinate"
    )


def test_static_overlap_is_quantised_by_float32_position_resolution(device):
    """Characterisation: the resting overlap is exact to a few ULPS, not to a
    few parts per million, and it is CONVERGED rather than noisy.

    Measured 2.588127e-7 m against an analytic 2.596590e-7 m — a 0.33% gap that
    is bit-identical at 20k, 40k, 60k, 80k and 120k steps. A drifting or
    unconverged solution would not repeat to the bit; a quantised one does.

    One ulp of a 2 mm coordinate is 2.33e-10 m, which is 0.09% of this overlap,
    so the gap is 3.6 quanta. This is the Block 10 cancellation finding showing
    up STATICALLY: overlap is a small difference of large numbers, and its
    absolute resolution is fixed by the coordinate magnitude no matter how long
    the run continues.

    Consequence for Validation Target 5: the target is mean delta/R < 1%, and
    this contact sits at 0.013% — a factor of 77 inside it. A 0.33% relative
    error on the overlap is therefore immaterial to the criterion. It would
    NOT be immaterial to a criterion phrased on the overlap itself, which is
    worth remembering when the Block 17 diagnostics report one.
    """
    state, solver, _ = _settle_on_floor(device, BASE)
    first = float(state.positions()[0][2])
    solver.run(60_000)
    second = float(state.positions()[0][2])

    assert first == second, "resting height was not bit-stable"
    overlap = RADIUS - first
    assert overlap / RADIUS < 0.01, "violates Validation Target 5"


# ── required test #4: the inclined plane ──────────────────────────────────────


def _slide_on_incline(device, material, angle_deg, steps=40_000,
                      lock_rotation=False):
    """Tilted gravity, flat floor. Returns (state, solver)."""
    theta = math.radians(angle_deg)
    gravity = (G * math.sin(theta), 0.0, -G * math.cos(theta))
    # Long domain: at 45 deg the particle covers tens of millimetres, and
    # running it into the +x wall would silently end the measurement.
    state, solver, _ = _on_floor(
        device, material, gravity, height=material.radius,
        lock_rotation=lock_rotation,
        bounds=((-1.0, -0.05, 0.0), (1.0, 0.05, 0.05)),
    )
    solver.run(steps)
    return state, solver


#: A SPHERE does not slide at the block friction angle atan(mu). Rolling
#: without slipping needs friction (2/7) m g sin(th) against an available
#: mu m g cos(th), so the contact point slips only when tan(th) > (7/2) mu.
#: Same 7/2 as the Block 13 oblique-impact threshold, and for the same reason:
#: the contact point of a sphere carries spin as well as translation.
SPHERE_SLIDE_ANGLE = math.degrees(math.atan(3.5 * BASE.friction_wall))


def test_block_and_sphere_sliding_thresholds_differ():
    """The distinction that a first draft of this file got wrong.

    A block on a slope slides above atan(mu) = 26.57 deg. A sphere with the same
    mu does not slide until 60.26 deg — between those two angles it rolls
    without slipping. Asserting the block criterion against a sphere is a
    category error, and it is the reason the first version of the slip test
    failed with a measured slip of 3e-5 m/s.
    """
    block = math.degrees(math.atan(BASE.friction_wall))
    assert block == pytest.approx(26.565, abs=1e-3)
    assert SPHERE_SLIDE_ANGLE == pytest.approx(60.255, abs=1e-3)
    assert SPHERE_SLIDE_ANGLE > block


def test_contact_point_slips_above_the_friction_angle(device):
    """REQUIRED TEST #4, part one.

    The observable is SLIP AT THE CONTACT POINT — v_x - omega_y*r for a sphere
    rolling along +x — not the centre velocity. A sphere below the friction
    angle still moves; what distinguishes the regimes is whether its surface
    slides against the wall.
    """
    state, _ = _slide_on_incline(device, BASE, SPHERE_SLIDE_ANGLE + 12.0)

    v_x = float(state.velocities()[0][0])
    omega_y = float(state.angular_velocities()[0][1])
    slip = v_x - omega_y * RADIUS
    assert v_x > 0.0, "particle did not move down the slope"
    assert abs(slip) > 0.1 * abs(v_x), (
        f"contact point is not slipping: v_x={v_x:.4f}, slip={slip:.4f}"
    )


def test_contact_point_does_not_slip_below_the_friction_angle(device):
    """REQUIRED TEST #4, part two: below the angle it rolls rather than slides.

    A frictionless-rolling sphere satisfies v = omega r exactly, and the
    acceleration is (5/7) g sin(theta) rather than g sin(theta) — the missing
    2/7 goes into spinning it up. Both are checked, because the rolling
    constraint alone would also be satisfied by a particle that never moved.
    """
    # 45 deg is well ABOVE the block friction angle of 26.57 and well below
    # the sphere threshold of 60.26 — the regime where the two criteria give
    # opposite answers, so it is the sharpest place to check.
    angle = 45.0
    theta = math.radians(angle)
    state, solver = _slide_on_incline(device, BASE, angle)

    v_x = float(state.velocities()[0][0])
    omega_y = float(state.angular_velocities()[0][1])
    slip = v_x - omega_y * RADIUS
    assert abs(slip) < 0.02 * abs(v_x), (
        f"contact point slipped below the friction angle: slip={slip:.5f}"
    )

    expected = (5.0 / 7.0) * G * math.sin(theta) * solver.time
    assert v_x == pytest.approx(expected, rel=0.05), (
        f"rolling acceleration {v_x / solver.time:.3f} vs (5/7) g sin(th) "
        f"= {(5.0 / 7.0) * G * math.sin(theta):.3f} m/s^2"
    )


def test_rolling_resistance_holds_a_sphere_on_a_shallow_slope(device):
    """REQUIRED TEST #4, part three: what "holds" actually requires.

    Sliding friction alone cannot hold a sphere at any angle, because rolling
    has no threshold. Rolling resistance does, and this is the test that
    justifies the Block 14 term physically rather than by assertion.
    """
    material = dataclasses.replace(BASE, rolling_friction=0.10)
    state, _ = _slide_on_incline(device, material, 2.0)

    v_x = float(state.velocities()[0][0])
    assert abs(v_x) < 1e-3, f"sphere crept down a 2 degree slope at {v_x:.4e} m/s"

    without = dataclasses.replace(BASE, rolling_friction=0.0)
    state_free, _ = _slide_on_incline(device, without, 2.0)
    assert float(state_free.velocities()[0][0]) > 1e-2, (
        "control case failed: the sphere should roll freely without resistance"
    )


# ── the Sunday block-sliding benchmark ────────────────────────────────────────


def test_sliding_block_travels_the_analytic_distance(device):
    """[Sunday2020] section 4: v0 = 5 m/s, mu = 0.5, distance = 2.5484 m.

    ROTATION IS LOCKED. The classical result d = v0^2 / (2 mu g) describes a
    BLOCK, and a sphere is not one — it would spin up and transition to rolling
    after a fraction of the distance. Suppressing rotation via infinite moment
    of inertia turns the sphere into the body the formula is about, rather than
    comparing a sphere against a result that was never meant for it.
    """
    material = dataclasses.replace(
        BASE, friction_wall=0.5, rolling_friction=0.0, restitution_wall=0.3
    )
    expected = 5.0**2 / (2.0 * 0.5 * G)
    assert expected == pytest.approx(2.5484, abs=1e-4)

    r = material.radius
    state = ParticleState.allocate(1, device, r, material.density)
    state.set_positions([[0.0, 0.0, r]])
    state.set_velocities([[5.0, 0.0, 0.0]])
    state.set_rotation_locked([0])

    budget = _budget(material, impact_velocity=5.0)
    boundary = BoxBoundary(state, material.wall_params(),
                           (-1.0, -1.0, 0.0), (5.0, 1.0, 1.0))
    solver = Solver(state, dt=budget.limit, gravity=(0.0, 0.0, -G), budget=budget,
                    boundary=boundary)

    for _ in range(4000):
        solver.run(200)
        if float(state.velocities()[0][0]) <= 0.0:
            break
    else:
        raise AssertionError("block never came to rest")

    travelled = float(state.positions()[0][0])
    assert travelled == pytest.approx(expected, rel=0.02), (
        f"travelled {travelled:.4f} m vs analytic {expected:.4f} m"
    )


def test_wall_history_is_cleared_when_a_particle_leaves_the_wall(device):
    """A slot whose contact has ended must not retain its displacement."""
    material = dataclasses.replace(BASE, friction_wall=0.5)
    state, solver, boundary = _on_floor(
        device, material, (0.0, 0.0, -G), height=RADIUS, velocity=(0.5, 0.0, 0.0)
    )
    solver.run(2000)
    assert np.abs(boundary.displacements()[0, 2]).max() > 0.0, "no history built"

    # Lift it clear of the floor.
    state.set_positions([[0.0, 0.0, 0.02]])
    state.set_velocities([[0.0, 0.0, 0.0]])
    solver.run(10)
    np.testing.assert_allclose(boundary.displacements()[0, 2], 0.0, atol=0.0)
