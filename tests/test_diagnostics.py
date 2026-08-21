"""Overlap diagnostics and drift bounds — Block 17.

Validation Target 5 has two halves. The stiffness-scaling half is Phase 3 work.
The half that lands here is the OVERLAP CHECK, which the target says must run on
every validation case:

    PASS: mean delta_n / radius < 1%, and max delta_n / radius < 5%

Excessive overlap is the failure mode soft contacts produce. Particles
interpenetrate, the bed behaves like a fluid, and the repose angle collapses.
The reason the target demands this check even when the repose angle looks fine
is that a plausible-looking angle is exactly what a too-soft bed produces on the
way to being wrong.

The statistics are computed on hand-built configurations first, where the
correct answer is known by construction, and only then on a real settled bed.
"""

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.contacts import ContactList
from warp_dem.diagnostics import OverlapStats, overlap_stats, position_drift_bound
from warp_dem.forces import ContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.precision import EPS
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget
from warp_dem.walls import BoxBoundary

RADIUS = 0.002
DENSITY = 2500.0

GLASS = MaterialProperties(
    name="glass_beads",
    density=DENSITY,
    radius=RADIUS,
    youngs_modulus=1.0e8,
    poisson_ratio=0.20,
    restitution=0.90,
    friction_particle=0.25,
    friction_wall=0.35,
    rolling_friction=0.01,
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


def _detected(device, positions, radius=RADIUS):
    state = ParticleState.allocate(len(positions), device, radius, DENSITY)
    state.set_positions(positions)
    contacts = ContactList(len(positions), device)
    contacts.detect(state)
    return state, contacts


# ── overlap statistics on known configurations ────────────────────────────────


def test_no_contacts_reports_zero_rather_than_dividing_by_zero(device):
    state, contacts = _detected(
        device, [[0.0, 0.0, 0.0], [10.0 * RADIUS, 0.0, 0.0]]
    )
    stats = overlap_stats(state, contacts)
    assert stats.contacts == 0
    assert stats.mean == 0.0
    assert stats.maximum == 0.0


def test_single_contact_overlap_is_measured_exactly(device):
    overlap = 0.004 * RADIUS
    state, contacts = _detected(
        device, [[0.0, 0.0, 0.0], [2.0 * RADIUS - overlap, 0.0, 0.0]]
    )
    stats = overlap_stats(state, contacts)

    assert stats.contacts == 1
    assert stats.mean == pytest.approx(0.004, rel=1e-3)
    assert stats.maximum == pytest.approx(0.004, rel=1e-3)


def test_mean_and_max_differ_when_overlaps_differ(device):
    """Three particles in a row with deliberately unequal overlaps."""
    shallow = 0.002 * RADIUS
    deep = 0.010 * RADIUS
    state, contacts = _detected(device, [
        [0.0, 0.0, 0.0],
        [2.0 * RADIUS - shallow, 0.0, 0.0],
        [4.0 * RADIUS - shallow - deep, 0.0, 0.0],
    ])
    stats = overlap_stats(state, contacts)

    assert stats.contacts == 2
    assert stats.maximum == pytest.approx(0.010, rel=1e-2)
    assert stats.mean == pytest.approx(0.006, rel=1e-2)


def test_ratio_is_taken_against_the_smaller_radius(device):
    """Polydisperse: the same absolute overlap hurts the small particle more."""
    radii = np.array([RADIUS, 3.0 * RADIUS], dtype=np.float32)
    overlap = 0.01 * RADIUS
    state = ParticleState.allocate(2, device, RADIUS, DENSITY)
    state.radius.assign(radii)
    state.set_positions([[0.0, 0.0, 0.0], [4.0 * RADIUS - overlap, 0.0, 0.0]])
    contacts = ContactList(2, device)
    contacts.detect(state)

    stats = overlap_stats(state, contacts)
    assert stats.contacts == 1
    # delta / min(r) = 0.01 R / R, not 0.01 R / 3R.
    assert stats.maximum == pytest.approx(0.01, rel=1e-2)


def test_wall_contacts_are_included_when_a_boundary_is_given(device):
    overlap = 0.006 * RADIUS
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)
    state.set_positions([[0.01, 0.01, RADIUS - overlap]])
    contacts = ContactList(1, device)
    contacts.detect(state)
    boundary = BoxBoundary(state, GLASS.wall_params(), (0.0, 0.0, 0.0),
                           (0.02, 0.02, 0.02))

    without = overlap_stats(state, contacts)
    with_walls = overlap_stats(state, contacts, boundary)

    assert without.contacts == 0
    assert with_walls.contacts == 1
    assert with_walls.maximum == pytest.approx(0.006, rel=1e-2)


def test_target_verdict_uses_the_pre_registered_limits():
    """1% mean and 5% max, from docs/validation_targets.md."""
    assert OverlapStats(mean=0.005, maximum=0.04, contacts=10).within_target
    assert not OverlapStats(mean=0.02, maximum=0.04, contacts=10).within_target
    assert not OverlapStats(mean=0.005, maximum=0.06, contacts=10).within_target
    assert OverlapStats(0.005, 0.04, 10).verdict == "PASS"
    assert OverlapStats(0.02, 0.04, 10).verdict == "FAIL"
    assert "PASS" in OverlapStats(0.005, 0.04, 10).describe()
    assert "FAIL" in OverlapStats(0.02, 0.04, 10).describe()


def test_zero_contacts_is_reported_as_not_applicable_rather_than_pass():
    """A run that never formed a bed must not be mistaken for a good one.

    With no contacts the inequality holds trivially, so a naive verdict reads
    PASS. Target 5 is a claim about a packing; with no packing there is no
    claim. This surfaced in the Block 18 smoke run, which reported PASS while
    every particle was still in free fall.
    """
    empty = OverlapStats(mean=0.0, maximum=0.0, contacts=0)
    assert empty.verdict == "n/a - no contacts"
    assert "PASS" not in empty.describe()


# ── Validation Target 5 on a real bed ─────────────────────────────────────────


def test_a_settled_bed_meets_validation_target_5(device):
    """The Target 5 overlap criterion, on a bed rather than a spreadsheet.

    Block 8 justified E = 1e8 Pa from a static-load table. This is the first
    time the criterion is checked against an actual simulated packing, which is
    what the target asks for — it says the check runs on every validation case,
    not that it is computed once from a formula.
    """
    n = 64
    r = GLASS.radius
    budget = compute_budget(
        r, GLASS.density, GLASS.youngs_modulus, GLASS.poisson_ratio,
        impact_velocity=1.0, hertz_steps=25,
    )

    rng = np.random.default_rng(0)
    grid = np.array(
        [(i, j, k) for k in range(4) for i in range(4) for j in range(4)],
        dtype=np.float64,
    )[:n] * (2.2 * r)
    grid[:, 2] += 3.0 * r
    grid[:, :2] += 2.0 * r
    grid += rng.uniform(-0.05 * r, 0.05 * r, size=grid.shape)

    state = ParticleState.allocate(n, device, r, GLASS.density)
    state.set_positions(grid)
    model = ContactModel(state, GLASS.pair_params())
    # Size the domain FROM the geometry — a hard-coded box that the grid
    # overruns starts particles buried in a wall, which the Block 17 drift
    # probe demonstrated by launching a bed at 12 m/s.
    boundary = BoxBoundary(
        state, GLASS.wall_params(), (0.0, 0.0, 0.0),
        (float(grid[:, 0].max()) + 2.0 * r,
         float(grid[:, 1].max()) + 2.0 * r,
         float(grid[:, 2].max()) + 10.0 * r),
    )
    solver = Solver(state, dt=budget.limit, gravity=(0.0, 0.0, -9.81),
                    budget=budget, contact_model=model, boundary=boundary)
    solver.run(30_000)

    stats = overlap_stats(state, model.contacts, boundary)
    assert stats.contacts > 20, f"bed did not form: {stats.describe()}"
    assert stats.within_target, stats.describe()
    # And comfortably, not marginally.
    assert stats.mean < 0.002, stats.describe()


# ── drift bound ───────────────────────────────────────────────────────────────


def test_drift_bound_scales_with_coordinate_and_step_count(device):
    """Both scalings from docs/precision.md, asserted rather than assumed."""
    state = ParticleState.allocate(1, device, RADIUS, DENSITY)

    state.set_positions([[0.1, 0.0, 0.0]])
    base = position_drift_bound(state, 1000)

    state.set_positions([[0.2, 0.0, 0.0]])
    assert position_drift_bound(state, 1000) == pytest.approx(2.0 * base)

    state.set_positions([[0.1, 0.0, 0.0]])
    assert position_drift_bound(state, 2000) == pytest.approx(2.0 * base)


def test_drift_bound_uses_the_largest_coordinate(device):
    """Rounding scales with ulp(x), so the worst-placed particle sets the bound."""
    state = ParticleState.allocate(3, device, RADIUS, DENSITY)
    state.set_positions([[0.01, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.02]])
    assert position_drift_bound(state, 1000) == pytest.approx(
        0.5 * EPS * 0.5 * 1000, rel=1e-6
    )
