"""Material properties and the derived constants the contact kernels consume.

Two layers, deliberately separated.

HOST SIDE: `MaterialProperties` mirrors a `configs/material/*.yaml` node. It
holds what a person can measure or look up — density, Young's modulus, Poisson
ratio, restitution, friction coefficients.

DEVICE SIDE: `ContactParams` is a `wp.struct` holding what a kernel actually
needs — effective moduli and a damping ratio. These are algebraic combinations
of the host values that do NOT vary per particle, so computing them once on the
host and passing a struct beats recomputing them per contact per step.

What is NOT in the struct: effective radius and effective mass. Those depend on
WHICH two particles are touching, so they are computed in-kernel from the
radius and mass arrays. That costs a few flops and buys polydispersity for
free — a monodisperse shortcut here would have to be unpicked in Phase 3.

Two instances are built from one material: particle-particle and
particle-wall. They differ in more than friction. A wall is rigid, so it
contributes nothing to the compliance sum, and the effective moduli are twice
the particle-particle values:

    1/E* = (1-nu_1^2)/E_1 + (1-nu_2^2)/E_2      1/G* = (2-nu_1)/G_1 + (2-nu_2)/G_2

    identical spheres   E* = E / (2(1-nu^2))     G* = G / (2(2-nu))
    sphere on rigid wall E* = E / (1-nu^2)       G* = G / (2-nu)

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects, and that applies to `wp.struct`
field annotations exactly as it does to kernel signatures.
"""

import math
from dataclasses import dataclass

import warp as wp

from warp_dem.precision import scalar
from warp_dem.timestep import effective_youngs_modulus, shear_modulus


@wp.struct
class ContactParams:
    """Per-material contact constants, uniform across every pair.

    Attributes:
        e_eff: effective Young's modulus E* [Pa].
        g_eff: effective shear modulus G* [Pa].
        beta:  Tsuji damping ratio, derived from restitution. NEGATIVE, or zero
               for a perfectly elastic contact. Kernels use its magnitude; the
               sign is preserved so that beta == 0.0 is unambiguous.
        mu_s:  Coulomb sliding friction coefficient.
        mu_r:  rolling friction coefficient.
    """

    e_eff: scalar
    g_eff: scalar
    beta: scalar
    mu_s: scalar
    mu_r: scalar


def damping_ratio(restitution: float) -> float:
    """Tsuji damping ratio beta from a target coefficient of restitution.

        beta = ln(e) / sqrt(ln^2(e) + pi^2)

    This is the closed-form inverse of the Hertzian collision problem: it is
    the value that makes a viscously damped Hertz contact rebound at exactly
    `e`. It is why the model reproduces its own input restitution rather than
    requiring a damping coefficient to be tuned per material.

    Returns a value in (-1, 0]. e = 1 gives exactly 0.0, i.e. no damping.

    Two properties worth knowing:

    - beta is DIMENSIONLESS and velocity-independent. The damping coefficient
      it feeds, c = 2 sqrt(5/6) |beta| sqrt(S_n m*), carries the velocity
      dependence through S_n ~ sqrt(delta). That is what makes measured
      restitution independent of impact speed, which Validation Target 4
      requires and which a linear (Hooke) contact model does NOT deliver.
    - e -> 0 sends beta -> -1, not to minus infinity. Fully plastic is a
      finite, well-behaved limit.
    """
    if not 0.0 < restitution <= 1.0:
        raise ValueError(
            f"restitution must lie in (0, 1], got {restitution}. "
            "e = 0 is perfectly plastic and has no finite damping ratio."
        )
    if restitution == 1.0:
        return 0.0
    ln_e = math.log(restitution)
    return ln_e / math.sqrt(ln_e * ln_e + math.pi * math.pi)


def effective_shear_modulus(youngs_modulus: float, poisson_ratio: float) -> float:
    """Mindlin effective shear modulus for two IDENTICAL spheres.

    1/G* = (2 - nu_1)/G_1 + (2 - nu_2)/G_2, which for identical bodies reduces
    to G* = G / (2(2 - nu)).
    """
    g = shear_modulus(youngs_modulus, poisson_ratio)
    return g / (2.0 * (2.0 - poisson_ratio))


def wall_youngs_modulus(youngs_modulus: float, poisson_ratio: float) -> float:
    """Hertz effective modulus for a sphere against a RIGID wall.

    The wall contributes no compliance, so only one term survives the sum:
    E* = E / (1 - nu^2), exactly twice the two-sphere value.
    """
    return youngs_modulus / (1.0 - poisson_ratio**2)


def wall_shear_modulus(youngs_modulus: float, poisson_ratio: float) -> float:
    """Mindlin effective shear modulus for a sphere against a RIGID wall."""
    return shear_modulus(youngs_modulus, poisson_ratio) / (2.0 - poisson_ratio)


@dataclass(frozen=True)
class MaterialProperties:
    """One material, as configured. Mirrors a configs/material/*.yaml node.

    Frozen because a material is an input to a run, not a thing a run mutates.
    Deriving a variant — the stiffness sweep in Validation Target 5, say — goes
    through `replace()`, which makes the variation explicit at the call site.
    """

    name: str
    density: float
    radius: float
    youngs_modulus: float
    poisson_ratio: float
    restitution: float
    friction_particle: float
    friction_wall: float
    rolling_friction: float
    restitution_wall: float

    def __post_init__(self) -> None:
        for field, value in (
            ("density", self.density),
            ("radius", self.radius),
            ("youngs_modulus", self.youngs_modulus),
        ):
            if value <= 0.0:
                raise ValueError(f"{field} must be positive, got {value}")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError(
                f"poisson_ratio must lie in (-1, 0.5), got {self.poisson_ratio}"
            )
        for field, value in (
            ("friction_particle", self.friction_particle),
            ("friction_wall", self.friction_wall),
            ("rolling_friction", self.rolling_friction),
        ):
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative, got {value}")
        # Validates both restitutions and raises with the right message.
        damping_ratio(self.restitution)
        damping_ratio(self.restitution_wall)

    @classmethod
    def from_config(cls, cfg) -> "MaterialProperties":
        """Build from a Hydra/OmegaConf material node.

        `restitution_wall` is optional and defaults to the particle-particle
        value. The two genuinely differ — Validation Target 1 quotes e_pp = 0.97
        and e_pw = 0.82 for glass beads — but defaulting keeps every existing
        material config valid rather than breaking them all at once.
        """
        contact = cfg.contact
        return cls(
            name=str(cfg.get("name", "unnamed")),
            density=float(cfg.density),
            radius=float(cfg.radius if "radius" in cfg else cfg.shape.radius),
            youngs_modulus=float(contact.youngs_modulus),
            poisson_ratio=float(contact.poisson_ratio),
            restitution=float(contact.restitution),
            friction_particle=float(contact.friction_particle),
            friction_wall=float(contact.friction_wall),
            rolling_friction=float(contact.rolling_friction),
            restitution_wall=float(
                contact.restitution_wall
                if "restitution_wall" in contact
                else contact.restitution
            ),
        )

    def pair_params(self) -> ContactParams:
        """Device-side constants for a particle-particle contact."""
        p = ContactParams()
        p.e_eff = effective_youngs_modulus(self.youngs_modulus, self.poisson_ratio)
        p.g_eff = effective_shear_modulus(self.youngs_modulus, self.poisson_ratio)
        p.beta = damping_ratio(self.restitution)
        p.mu_s = self.friction_particle
        p.mu_r = self.rolling_friction
        return p

    def wall_params(self) -> ContactParams:
        """Device-side constants for a particle-wall contact.

        Stiffer than particle-particle by exactly 2x in both moduli, because a
        rigid wall adds no compliance to the series sum.
        """
        p = ContactParams()
        p.e_eff = wall_youngs_modulus(self.youngs_modulus, self.poisson_ratio)
        p.g_eff = wall_shear_modulus(self.youngs_modulus, self.poisson_ratio)
        p.beta = damping_ratio(self.restitution_wall)
        p.mu_s = self.friction_wall
        p.mu_r = self.rolling_friction
        return p
