"""Timestep stability bound tests.

Structured as scaling laws rather than magic numbers wherever possible: a
scaling test survives a change of convention (rayleigh_fraction, hertz_steps)
and pins the physics, whereas a hard-coded microsecond value pins an arithmetic
result and breaks the moment a default moves.

Two regression values are pinned deliberately, against real soda-lime glass, so
that an error in the formula itself cannot pass silently.
"""

import math

import pytest

from warp_dem import (
    TimestepError,
    assert_timestep_valid,
    compute_budget,
    hertz_static_overlap,
)
from warp_dem.timestep import (
    effective_youngs_modulus,
    hertz_contact_time,
    rayleigh_time,
    shear_modulus,
    sphere_mass,
)

GLASS = dict(radius=0.002, density=2500.0, poisson_ratio=0.20)
E_REAL = 6.8e10


def test_shear_modulus_relation():
    assert shear_modulus(1.0e9, 0.25) == pytest.approx(1.0e9 / 2.5)


def test_effective_modulus_of_identical_spheres():
    e, nu = 1.0e9, 0.3
    assert effective_youngs_modulus(e, nu) == pytest.approx(e / (2 * (1 - nu**2)))


def test_rayleigh_time_scales_as_inverse_sqrt_stiffness():
    a = rayleigh_time(youngs_modulus=1.0e8, **GLASS)
    b = rayleigh_time(youngs_modulus=4.0e8, **GLASS)
    assert a / b == pytest.approx(2.0, rel=1e-6)


def test_rayleigh_time_is_linear_in_radius():
    a = rayleigh_time(radius=0.002, density=2500.0, youngs_modulus=1e8, poisson_ratio=0.2)
    b = rayleigh_time(radius=0.004, density=2500.0, youngs_modulus=1e8, poisson_ratio=0.2)
    assert b / a == pytest.approx(2.0, rel=1e-6)


def test_hertz_time_scales_as_stiffness_to_minus_two_fifths():
    kw = dict(**GLASS, impact_velocity=1.0)
    a = hertz_contact_time(youngs_modulus=1.0e8, **kw)
    b = hertz_contact_time(youngs_modulus=1.0e9, **kw)
    assert a / b == pytest.approx(10.0**0.4, rel=1e-6)


def test_hertz_time_is_linear_in_radius():
    """m ~ R^3, so (m^2 / R)^(1/5) ~ (R^6 / R)^(1/5) = R. Both bounds are
    linear in R, which is why coarse-graining is such an effective speedup."""
    kw = dict(density=2500.0, youngs_modulus=1e8, poisson_ratio=0.2, impact_velocity=1.0)
    a = hertz_contact_time(radius=0.002, **kw)
    b = hertz_contact_time(radius=0.006, **kw)
    assert b / a == pytest.approx(3.0, rel=1e-6)


def test_faster_impacts_are_shorter():
    kw = dict(**GLASS, youngs_modulus=1.0e8)
    slow = hertz_contact_time(impact_velocity=0.1, **kw)
    fast = hertz_contact_time(impact_velocity=10.0, **kw)
    assert fast < slow
    assert slow / fast == pytest.approx(100.0**0.2, rel=1e-6)


def test_zero_impact_velocity_is_rejected():
    with pytest.raises(ValueError):
        hertz_contact_time(**GLASS, youngs_modulus=1e8, impact_velocity=0.0)


def test_real_glass_regression():
    """Pinned against hand-computed values for soda-lime glass. Guards the
    formulae themselves, which no scaling test can do."""
    t_r = rayleigh_time(youngs_modulus=E_REAL, **GLASS)
    assert t_r == pytest.approx(2.053e-6, rel=1e-3)

    t_c = hertz_contact_time(youngs_modulus=E_REAL, impact_velocity=1.0, **GLASS)
    assert t_c == pytest.approx(12.2e-6, rel=1e-2)


def test_binding_constraint_flips_with_stiffness():
    """Rayleigh binds for stiff materials; Hertz takes over once softened.
    The code must report which, because the remedies differ."""
    assert compute_budget(youngs_modulus=E_REAL, **GLASS).binding == "Rayleigh"
    assert compute_budget(youngs_modulus=1.0e7, **GLASS).binding == "Hertz"


def test_limit_is_the_smaller_bound():
    b = compute_budget(youngs_modulus=1.0e8, **GLASS)
    assert b.limit == min(b.rayleigh_limit, b.hertz_limit)
    assert b.limit <= b.rayleigh_limit
    assert b.limit <= b.hertz_limit


def test_softening_buys_sqrt_of_the_stiffness_ratio_at_the_rayleigh_bound():
    """The central justification for stiffness scaling, asserted as physics."""
    stiff = compute_budget(youngs_modulus=1.0e10, **GLASS)
    soft = compute_budget(youngs_modulus=1.0e8, **GLASS)
    assert soft.rayleigh_limit / stiff.rayleigh_limit == pytest.approx(10.0, rel=1e-6)


def test_valid_timestep_passes():
    b = compute_budget(youngs_modulus=1.0e8, **GLASS)
    assert_timestep_valid(b.limit * 0.5, b)
    assert_timestep_valid(b.limit, b)


def test_oversized_timestep_raises_with_an_actionable_message():
    b = compute_budget(youngs_modulus=1.0e8, **GLASS)
    with pytest.raises(TimestepError) as exc:
        assert_timestep_valid(b.limit * 3.0, b)

    message = str(exc.value)
    assert b.binding in message
    assert "stiffness scaling" in message
    assert "Validation Target 5" in message
    assert f"{b.limit:.3e}" in message


def test_nonpositive_timestep_raises():
    b = compute_budget(youngs_modulus=1.0e8, **GLASS)
    with pytest.raises(TimestepError):
        assert_timestep_valid(0.0, b)


def test_bad_material_parameters_are_rejected():
    with pytest.raises(ValueError):
        compute_budget(radius=0.0, density=2500.0, youngs_modulus=1e8, poisson_ratio=0.2)
    with pytest.raises(ValueError):
        compute_budget(radius=0.002, density=2500.0, youngs_modulus=1e8, poisson_ratio=0.6)


def test_static_overlap_meets_target_five_for_a_deep_bed():
    """Validation Target 5: delta/R < 1% under the deepest expected load.

    Fifty particle weights is roughly a repose pile; it is the load that
    matters, not a single contact. This is the criterion that condemned the
    original E = 5 MPa and set the current value.
    """
    radius = GLASS["radius"]
    load = 50 * sphere_mass(radius, GLASS["density"]) * 9.81

    delta = hertz_static_overlap(load, radius, 1.0e8, GLASS["poisson_ratio"])
    assert delta / radius < 0.01

    too_soft = hertz_static_overlap(load, radius, 5.0e6, GLASS["poisson_ratio"])
    assert too_soft / radius > 0.01


def test_overlap_grows_as_load_to_the_two_thirds():
    kw = dict(radius=0.002, youngs_modulus=1e8, poisson_ratio=0.2)
    a = hertz_static_overlap(1.0, **kw)
    b = hertz_static_overlap(8.0, **kw)
    assert b / a == pytest.approx(4.0, rel=1e-6)


def test_solver_rejects_an_unstable_timestep():
    """The Block 8 done-when criterion, at the point of use."""
    import numpy as np

    from warp_dem import resolve_device
    from warp_dem.solver import Solver
    from warp_dem.state import ParticleState

    device = resolve_device("cpu")
    budget = compute_budget(youngs_modulus=1.0e8, **GLASS)
    state = ParticleState.allocate(1, device, GLASS["radius"], GLASS["density"])

    with pytest.raises(TimestepError):
        Solver(state, dt=budget.limit * 10.0, budget=budget)

    solver = Solver(state, dt=budget.limit * 0.5, budget=budget)
    assert solver.budget is budget
    assert math.isfinite(np.sum(state.positions()))