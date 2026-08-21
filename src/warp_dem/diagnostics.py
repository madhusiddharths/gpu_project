"""Energy and overlap diagnostics — Blocks 16 and 17.

Two jobs, both device-side reductions.

ENERGY (Block 16) exists to audit the integrator. A frictionless, undamped
configuration must conserve total energy, and "total" has three parts:

    E = sum 1/2 m v^2 + 1/2 I omega^2       kinetic, translational and rotational
      + sum -m (g . x)                       gravitational potential
      + sum (8/15) E* sqrt(R*) delta^2.5     ELASTIC, stored inside the contacts

The third term is the one that is easy to forget and fatal to omit. At any
instant some of the system's energy is inside the springs, and a "total" that
counts only kinetic plus potential oscillates violently every time anything
touches anything. That is not an integrator failure; it is an accounting failure,
and it looks identical from the outside.

OVERLAP (Block 17) exists to police Validation Target 5: mean delta/R below 1%
and max below 5%. Excessive overlap is the failure mode soft contacts produce —
particles interpenetrate, the bed behaves like a fluid, and the repose angle
collapses. The overlap check catches that even when the repose angle happens to
look plausible, which is why the target requires it on every validation case
rather than only when something looks wrong.

NON-DETERMINISM WARNING. These reductions use wp.atomic_add on floats, and
floating-point addition is not associative, so the sum depends on the order the
atomics happen to retire. On CUDA that order varies between runs and the last
digits of a reported energy will vary with it. That is acceptable for a
diagnostic and would NOT be acceptable for anything the solver consumed — it is
the same reason parity tests compare within a tolerance rather than bit-exactly.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from warp_dem.contacts import ContactList
from warp_dem.forces import (
    MIN_SEPARATION,
    effective_radius,
    hertz_elastic_energy,
)
from warp_dem.precision import scalar, vec3
from warp_dem.state import ParticleState
from warp_dem.walls import WALL_COUNT


@wp.kernel
def accumulate_kinetic_energy(
    vel: wp.array(dtype=vec3),
    mass: wp.array(dtype=scalar),
    omega: wp.array(dtype=vec3),
    inertia: wp.array(dtype=scalar),
    out: wp.array(dtype=scalar),
):
    """1/2 m v^2 + 1/2 I omega^2, summed over particles.

    Rotational kinetic energy is included because it is REAL energy that the
    tangential force moves in and out of. Omitting it would make a frictional
    run appear to lose energy exactly when it was being converted to spin.
    """
    i = wp.tid()
    v = vel[i]
    w = omega[i]
    wp.atomic_add(
        out, 0, 0.5 * mass[i] * wp.dot(v, v) + 0.5 * inertia[i] * wp.dot(w, w)
    )


@wp.kernel
def accumulate_potential_energy(
    pos: wp.array(dtype=vec3),
    mass: wp.array(dtype=scalar),
    gravity: vec3,
    out: wp.array(dtype=scalar),
):
    """-m (g . x). The datum is the coordinate origin, so this can be negative;
    only CHANGES in it are physical."""
    i = wp.tid()
    wp.atomic_add(out, 0, -mass[i] * wp.dot(gravity, pos[i]))


@wp.kernel
def accumulate_elastic_energy(
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    e_eff: scalar,
    out: wp.array(dtype=scalar),
):
    """Energy stored in the Hertz springs, summed over active contacts."""
    p = wp.tid()
    if p >= pair_count[0]:
        return

    i = pair_i[p]
    j = pair_j[p]
    d = pos[j] - pos[i]
    dist = wp.length(d)
    if dist < MIN_SEPARATION:
        return
    delta = radius[i] + radius[j] - dist
    if delta <= 0.0:
        return

    wp.atomic_add(
        out, 0, hertz_elastic_energy(delta, effective_radius(radius[i], radius[j]), e_eff)
    )


@wp.kernel
def accumulate_overlap_stats(
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    total: wp.array(dtype=scalar),
    peak: wp.array(dtype=scalar),
    count: wp.array(dtype=wp.int32),
):
    """Sum, maximum and count of delta/radius over active particle contacts.

    The ratio is taken against the SMALLER of the two radii. For a monodisperse
    bed that is just the radius; for a polydisperse one it is the conservative
    choice, because the same absolute overlap is a larger fraction of the
    smaller particle and Validation Target 5 is a statement about how badly any
    particle is interpenetrating.
    """
    p = wp.tid()
    if p >= pair_count[0]:
        return

    i = pair_i[p]
    j = pair_j[p]
    d = pos[j] - pos[i]
    dist = wp.length(d)
    if dist < MIN_SEPARATION:
        return
    ri = radius[i]
    rj = radius[j]
    delta = ri + rj - dist
    if delta <= 0.0:
        return

    ratio = delta / wp.min(ri, rj)
    wp.atomic_add(total, 0, ratio)
    wp.atomic_max(peak, 0, ratio)
    wp.atomic_add(count, 0, 1)


@wp.kernel
def accumulate_wall_overlap_stats(
    pos: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    bounds_min: vec3,
    bounds_max: vec3,
    total: wp.array(dtype=scalar),
    peak: wp.array(dtype=scalar),
    count: wp.array(dtype=wp.int32),
):
    """Same statistics for the six walls.

    Wall contacts are counted separately because they are TWICE as stiff, so a
    given load produces a smaller overlap there. Pooling them with particle
    contacts would flatter the mean, and Target 5 is a claim about the bed.
    """
    i = wp.tid()
    xi = pos[i]
    ri = radius[i]

    for w in range(WALL_COUNT):
        axis = w % 3
        dist = scalar(0.0)
        if w < 3:
            dist = xi[axis] - bounds_min[axis]
        else:
            dist = bounds_max[axis] - xi[axis]

        delta = ri - dist
        if delta > 0.0 and dist > -ri:
            ratio = delta / ri
            wp.atomic_add(total, 0, ratio)
            wp.atomic_max(peak, 0, ratio)
            wp.atomic_add(count, 0, 1)


@dataclass(frozen=True)
class EnergyBreakdown:
    """The three parts of the total, kept separate deliberately.

    A single total tells you conservation failed; the split tells you where. A
    kinetic term that grows while elastic shrinks is a contact injecting energy;
    both growing while potential falls is just gravity doing work.
    """

    kinetic: float
    potential: float
    elastic: float

    @property
    def total(self) -> float:
        return self.kinetic + self.potential + self.elastic

    def describe(self) -> str:
        return (
            f"kinetic {self.kinetic:12.6e} J   "
            f"potential {self.potential:12.6e} J   "
            f"elastic {self.elastic:12.6e} J   "
            f"total {self.total:12.6e} J"
        )


@dataclass(frozen=True)
class OverlapStats:
    """delta/radius statistics, the Validation Target 5 observable."""

    mean: float
    maximum: float
    contacts: int

    #: Target 5, pre-registered in docs/validation_targets.md before any solver
    #: code existed.
    MEAN_LIMIT = 0.01
    MAX_LIMIT = 0.05

    @property
    def within_target(self) -> bool:
        return self.mean < self.MEAN_LIMIT and self.maximum < self.MAX_LIMIT

    @property
    def verdict(self) -> str:
        """PASS, FAIL, or an explicit statement that there was nothing to check.

        A configuration with no contacts satisfies the inequality trivially, and
        reporting that as PASS is how a run that never formed a bed gets mistaken
        for a run that formed a good one. Target 5 is a claim about a packing; if
        there is no packing there is no claim, and the report says so.
        """
        if self.contacts == 0:
            return "n/a - no contacts"
        return "PASS" if self.within_target else "FAIL"

    def describe(self) -> str:
        return (
            f"contacts {self.contacts:8d}   "
            f"mean delta/R {self.mean:8.4%}   "
            f"max delta/R {self.maximum:8.4%}   "
            f"[Target 5: {self.verdict}]"
        )


def _reduce(kernel, dim, inputs, device) -> float:
    out = wp.zeros(1, dtype=scalar, device=device)
    wp.launch(kernel, dim=dim, inputs=[*inputs, out], device=device)
    return float(out.numpy()[0])


def kinetic_energy(state: ParticleState) -> float:
    return _reduce(
        accumulate_kinetic_energy,
        state.n,
        [state.vel, state.mass, state.omega, state.inertia],
        state.device,
    )


def potential_energy(state: ParticleState, gravity) -> float:
    g = gravity if isinstance(gravity, vec3) else vec3(*(float(v) for v in gravity))
    return _reduce(
        accumulate_potential_energy, state.n, [state.pos, state.mass, g], state.device
    )


def elastic_energy(state: ParticleState, contacts: ContactList, e_eff: float) -> float:
    """Requires `contacts` to have been detected at the CURRENT positions."""
    return _reduce(
        accumulate_elastic_energy,
        contacts.capacity,
        [
            contacts.pair_i, contacts.pair_j, contacts.pair_count,
            state.pos, state.radius, float(e_eff),
        ],
        state.device,
    )


def energy_breakdown(state: ParticleState, gravity, contacts=None,
                     e_eff: float = 0.0) -> EnergyBreakdown:
    """Full accounting. Pass `contacts` whenever contact forces are active.

    Omitting the elastic term does not merely lose a little accuracy — it makes
    the reported total oscillate at the contact frequency, which reads exactly
    like an unstable integrator.
    """
    stored = 0.0
    if contacts is not None and e_eff > 0.0:
        stored = elastic_energy(state, contacts, e_eff)
    return EnergyBreakdown(
        kinetic=kinetic_energy(state),
        potential=potential_energy(state, gravity),
        elastic=stored,
    )


def overlap_stats(state: ParticleState, contacts: ContactList,
                  boundary=None) -> OverlapStats:
    """Validation Target 5 statistics over particle and, optionally, wall contacts."""
    device = state.device
    total = wp.zeros(1, dtype=scalar, device=device)
    peak = wp.zeros(1, dtype=scalar, device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        accumulate_overlap_stats,
        dim=contacts.capacity,
        inputs=[
            contacts.pair_i, contacts.pair_j, contacts.pair_count,
            state.pos, state.radius, total, peak, count,
        ],
        device=device,
    )
    if boundary is not None:
        wp.launch(
            accumulate_wall_overlap_stats,
            dim=state.n,
            inputs=[
                state.pos, state.radius,
                boundary.bounds_min, boundary.bounds_max,
                total, peak, count,
            ],
            device=device,
        )

    n_contacts = int(count.numpy()[0])
    if n_contacts == 0:
        return OverlapStats(mean=0.0, maximum=0.0, contacts=0)
    return OverlapStats(
        mean=float(total.numpy()[0]) / n_contacts,
        maximum=float(peak.numpy()[0]),
        contacts=n_contacts,
    )


def position_drift_bound(state: ParticleState, additions: int) -> float:
    """Rigorous upper bound on float32 position drift for this configuration.

    Reported alongside overlap because the two compete: Block 6 measured drift
    growing linearly with step count when the velocity increment is nearly
    constant, and docs/precision.md notes that the worst case in drum geometry
    is 8.9 mm against a target contact overlap of ~20 um. If drift ever
    approaches the overlap, the contacts are being resolved into noise.

    See scripts/drift_probe.py for the MEASURED counterpart. This is the bound;
    that is the observation, and §5.1 of the handoff wants both.
    """
    from warp_dem.precision import accumulation_bound

    magnitude = float(np.abs(state.positions()).max()) if state.n else 0.0
    return accumulation_bound(magnitude, additions)
