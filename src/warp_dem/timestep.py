"""Stability bounds on the DEM timestep.

Two independent criteria, both upper bounds; the smaller one governs.

RAYLEIGH WAVE. A contact disturbance travels across a particle surface as a
Rayleigh (surface elastic) wave. The timestep must be short enough that the
wave cannot traverse a particle in one step.

    t_R = pi * R * sqrt(rho / G) / (0.1631 * nu + 0.8766)

The denominator is an empirical fit for Rayleigh wave speed as a fraction of
shear wave speed; it varies only over ~0.88-0.92 across all physical Poisson
ratios, so it is nearly a constant. Practitioners use a fraction of t_R,
conventionally 0.2.

HERTZ CONTACT DURATION. The Rayleigh criterion is velocity-blind. A separate
bound comes from how long a binary Hertzian collision actually lasts:

    t_c = 2.87 * (m_eff^2 / (R_eff * E_eff^2 * v_rel))^(1/5)

and one demands ~20-50 steps within that collision. Faster impacts are SHORTER
impacts, so a violent system needs a smaller timestep than a quiescent one.

Scaling behaviour, worth knowing by heart because it is what makes the project
feasible:

    t_R ~ R,  rho^(1/2),  E^(-1/2)
    t_c ~ R,  rho^(2/5),  E^(-2/5),  v^(-1/5)

Softening stiffness by a factor S therefore buys ~sqrt(S) in timestep. That is
stiffness scaling. It is legitimate ONLY if the reported quantities are shown
insensitive to it — see docs/timestep_and_stiffness.md and Validation Target 5.

WARNING - constants pending primary sources. The 0.1631/0.8766 Rayleigh fit and
the 2.87 Hertz prefactor are quoted widely in the DEM literature and in
commercial solver documentation. They are not yet confirmed here against the
original papers, in the same way Beverloo's constants are flagged in
docs/validation_targets.md. Confirm before Phase 3 reports against them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class TimestepError(ValueError):
    """Raised when a configured timestep violates a stability bound."""


def shear_modulus(youngs_modulus: float, poisson_ratio: float) -> float:
    """G = E / (2(1 + nu))."""
    return youngs_modulus / (2.0 * (1.0 + poisson_ratio))


def effective_youngs_modulus(youngs_modulus: float, poisson_ratio: float) -> float:
    """Hertz effective modulus for two IDENTICAL spheres.

    1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2, which for identical bodies reduces
    to E* = E / (2(1 - nu^2)).
    """
    return youngs_modulus / (2.0 * (1.0 - poisson_ratio**2))


def sphere_mass(radius: float, density: float) -> float:
    return density * (4.0 / 3.0) * math.pi * radius**3


def rayleigh_time(radius: float, density: float, youngs_modulus: float,
                  poisson_ratio: float) -> float:
    """Rayleigh wave traversal time for one particle, in seconds."""
    g = shear_modulus(youngs_modulus, poisson_ratio)
    return math.pi * radius * math.sqrt(density / g) / (0.1631 * poisson_ratio + 0.8766)


def hertz_contact_time(radius: float, density: float, youngs_modulus: float,
                       poisson_ratio: float, impact_velocity: float) -> float:
    """Duration of a Hertzian collision between two identical spheres."""
    if impact_velocity <= 0.0:
        raise ValueError("impact_velocity must be positive; a zero-speed impact "
                         "has no finite duration and imposes no bound")
    m_eff = sphere_mass(radius, density) / 2.0
    r_eff = radius / 2.0
    e_eff = effective_youngs_modulus(youngs_modulus, poisson_ratio)
    return 2.87 * (m_eff**2 / (r_eff * e_eff**2 * impact_velocity)) ** 0.2


def hertz_static_overlap(force: float, radius: float, youngs_modulus: float,
                         poisson_ratio: float) -> float:
    """Overlap at which Hertz repulsion balances a given load, in metres.

    Inverts F = (4/3) E* sqrt(R*) delta^1.5. Used to check that stiffness
    scaling has not made the particles measurably soft — the criterion in
    Validation Target 5 is delta/R < 1%.
    """
    r_eff = radius / 2.0
    e_eff = effective_youngs_modulus(youngs_modulus, poisson_ratio)
    return (force / ((4.0 / 3.0) * e_eff * math.sqrt(r_eff))) ** (2.0 / 3.0)


@dataclass(frozen=True)
class TimestepBudget:
    """The stability bounds for one material, and which of them governs."""

    rayleigh_time: float
    hertz_time: float
    rayleigh_limit: float
    hertz_limit: float
    impact_velocity: float
    rayleigh_fraction: float
    hertz_steps: int

    @property
    def limit(self) -> float:
        return min(self.rayleigh_limit, self.hertz_limit)

    @property
    def binding(self) -> str:
        return "Rayleigh" if self.rayleigh_limit <= self.hertz_limit else "Hertz"

    def describe(self) -> str:
        return (
            f"Rayleigh time      {self.rayleigh_time * 1e6:10.3f} us  "
            f"-> limit {self.rayleigh_limit * 1e6:8.3f} us "
            f"({self.rayleigh_fraction:.2f} x t_R)\n"
            f"Hertz contact time {self.hertz_time * 1e6:10.3f} us  "
            f"-> limit {self.hertz_limit * 1e6:8.3f} us "
            f"(t_c / {self.hertz_steps}, at {self.impact_velocity:g} m/s)\n"
            f"Binding constraint: {self.binding}\n"
            f"Maximum stable dt : {self.limit:.3e} s  ({self.limit * 1e6:.3f} us)"
        )


def compute_budget(radius: float, density: float, youngs_modulus: float,
                   poisson_ratio: float, impact_velocity: float = 1.0,
                   rayleigh_fraction: float = 0.2,
                   hertz_steps: int = 25) -> TimestepBudget:
    """Both bounds for a monodisperse material.

    Args:
        impact_velocity: the largest relative impact speed expected. For a
            rotating drum, roughly the free-fall speed from the cascade height.
            Assuming too low a value silently inflates the allowed timestep, so
            it is a required judgement, not a detail.
        rayleigh_fraction: conventionally 0.2.
        hertz_steps: timesteps demanded within one collision, conventionally
            20-50. Restitution accuracy is what this buys; 25 is a reasonable
            default and is revisited in Block 10.
    """
    for name, value in (("radius", radius), ("density", density),
                        ("youngs_modulus", youngs_modulus)):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError(f"poisson_ratio must lie in (-1, 0.5), got {poisson_ratio}")

    t_r = rayleigh_time(radius, density, youngs_modulus, poisson_ratio)
    t_c = hertz_contact_time(radius, density, youngs_modulus, poisson_ratio,
                             impact_velocity)
    return TimestepBudget(
        rayleigh_time=t_r,
        hertz_time=t_c,
        rayleigh_limit=rayleigh_fraction * t_r,
        hertz_limit=t_c / hertz_steps,
        impact_velocity=impact_velocity,
        rayleigh_fraction=rayleigh_fraction,
        hertz_steps=hertz_steps,
    )


def assert_timestep_valid(dt: float, budget: TimestepBudget) -> None:
    """Fail loudly, and actionably, on an unstable timestep.

    A silently unstable DEM run does not crash — it produces a plausible-looking
    bed that is quietly wrong, which is far worse than an exception. The message
    names the binding constraint and the available remedies, because "timestep
    too large" without them sends you guessing.
    """
    if dt <= 0.0:
        raise TimestepError(f"dt must be positive, got {dt}")
    if dt <= budget.limit:
        return

    raise TimestepError(
        f"Timestep {dt:.3e} s exceeds the stability limit {budget.limit:.3e} s "
        f"by {dt / budget.limit:.2f}x.\n"
        f"Binding constraint: {budget.binding}.\n"
        f"{budget.describe()}\n"
        f"Remedies, in order of preference:\n"
        f"  1. reduce dt to <= {budget.limit:.3e} s\n"
        f"  2. reduce youngs_modulus  (t_R ~ E^-1/2, t_c ~ E^-2/5) -- this is\n"
        f"     stiffness scaling; it must be validated, not assumed. Check the\n"
        f"     resulting static overlap against Validation Target 5 (delta/R < 1%).\n"
        f"  3. raise impact_velocity only if {budget.impact_velocity:g} m/s is\n"
        f"     pessimistic -- lowering it inflates the Hertz limit and is the\n"
        f"     easiest way to fool yourself\n"
        f"  4. increase particle radius (both bounds are linear in R)"
    )