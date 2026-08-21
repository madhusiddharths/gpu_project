"""Measure coefficient of restitution against its configured value.

Answers the two questions Block 10 exists to settle with data rather than
convention:

  1. How many timesteps inside a collision does Validation Target 4 actually
     need? `timestep.compute_budget` defaults to 25 because that is what the
     literature quotes. This measures it.
  2. Is measured restitution independent of impact speed? Target 4 requires a
     spread under 2% across 0.1-2.0 m/s, and it is the property that separates
     a Hertzian contact from a linear-spring one.

    python scripts/restitution.py
"""

import dataclasses

from warp_dem import describe_device, resolve_device
from warp_dem.forces import NormalContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget

GLASS = MaterialProperties(
    name="glass_beads",
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

RESTITUTIONS = (0.5, 0.7, 0.9, 0.97)
STEPS_PER_COLLISION = (10, 15, 25, 50, 100)
VELOCITIES = (0.1, 0.5, 1.0, 2.0)


def measure(device, material, v_approach, steps_per_collision):
    """One head-on collision. Returns (measured_e, steps_actually_in_contact)."""
    budget = compute_budget(
        material.radius, material.density, material.youngs_modulus,
        material.poisson_ratio, impact_velocity=v_approach,
        hertz_steps=steps_per_collision,
    )

    # budget.limit, not budget.hertz_limit: the Rayleigh bound can be the
    # smaller of the two, and asking for few steps per collision does not
    # entitle you to exceed it. At E = 1e8 Pa the Rayleigh bound clamps the
    # timestep from spc = 15 downward, so the requested figure and the
    # delivered one part company there — which is itself worth reporting.
    dt = budget.limit

    state = ParticleState.allocate(2, device, material.radius, material.density)
    r = material.radius
    state.set_positions([[-r, 0.0, 0.0], [r, 0.0, 0.0]])
    state.set_velocities([[v_approach / 2, 0.0, 0.0], [-v_approach / 2, 0.0, 0.0]])

    solver = Solver(
        state, dt=dt, gravity=(0.0, 0.0, 0.0), budget=budget,
        contact_model=NormalContactModel(state, material.pair_params()),
    )

    in_contact = 0
    touched = False
    for _ in range(4000):
        solver.step()
        gap = float(state.positions()[1][0] - state.positions()[0][0])
        if gap < 2.0 * r:
            in_contact += 1
            touched = True
        elif touched:
            break

    v = state.velocities()
    return abs(float(v[0][0] - v[1][0])) / v_approach, in_contact


def main() -> None:
    device = resolve_device("auto")
    print(describe_device(device))

    print("\n=== Relative error in restitution vs steps per collision ===")
    print("(Validation Target 4a demands < 2%)\n")
    header = f"{'e_in':>6}" + "".join(f"{f'spc={s}':>10}" for s in STEPS_PER_COLLISION)
    print(header)
    print("-" * len(header))
    for e in RESTITUTIONS:
        material = dataclasses.replace(GLASS, restitution=e)
        cells = []
        for spc in STEPS_PER_COLLISION:
            measured, _ = measure(device, material, 1.0, spc)
            cells.append(f"{100 * abs(measured - e) / e:9.3f}%")
        print(f"{e:6.2f}" + "".join(cells))

    print("\n=== Steps actually spent inside the contact ===")
    print("The 2.87 prefactor predicts the collision duration; this is what was")
    print("delivered. A systematic shortfall means the budget is optimistic.\n")
    print(f"{'requested':>10}{'actual':>10}{'ratio':>10}")
    print("-" * 30)
    for spc in STEPS_PER_COLLISION:
        _, actual = measure(device, GLASS, 1.0, spc)
        print(f"{spc:10d}{actual:10d}{actual / spc:10.2f}")

    print("\n=== Velocity independence (Validation Target 4b) ===")
    print("Spread must stay under 2% across 0.1-2.0 m/s.\n")
    head = f"{'e_in':>6}" + "".join(f"{v:>10}" for v in VELOCITIES) + f"{'spread':>10}"
    print(head)
    print("-" * len(head))
    for e in RESTITUTIONS:
        material = dataclasses.replace(GLASS, restitution=e)
        measured = [measure(device, material, v, 25)[0] for v in VELOCITIES]
        spread = (max(measured) - min(measured)) / (sum(measured) / len(measured))
        print(f"{e:6.2f}" + "".join(f"{m:10.5f}" for m in measured)
              + f"{100 * spread:9.3f}%")


if __name__ == "__main__":
    main()
