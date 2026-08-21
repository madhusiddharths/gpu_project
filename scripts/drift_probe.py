"""Measure float32 position drift, and settle the question §5.1 leaves open.

docs/precision.md establishes that `x += v*dt` in float32 drifts, and that the
severity depends entirely on whether successive roundings are CORRELATED:

    fully correlated (worst case)  drift ~ 0.5 EPS |x| N          8.9 mm
    fully decorrelated             drift ~ 0.5 EPS |x| sqrt(N)    8.9 um

against a contact overlap of ~20 um in drum geometry. The first is fatal, the
second is irrelevant, and the note says plainly that "a tumbling bed
decorrelates" is an argument rather than a measurement.

This script measures it. Two legs:

  1. BALLISTIC — a single particle at constant velocity, float32 device against
     a float64 host reference. Reproduces the Block 6 result and confirms drift
     grows LINEARLY in step count when the increment never changes sign.

  2. BED — a real settling bed, where velocities reverse constantly. Rather than
     re-running in double precision (see the note below), it measures the
     CORRELATION LENGTH directly: the mean number of steps a velocity component
     keeps its sign. If roundings are correlated in runs of L steps, the drift
     is a random walk over N/L runs of size L*delta:

         drift ~ delta * L * sqrt(N/L) = delta * sqrt(L*N)

     so the enhancement over the fully decorrelated estimate is sqrt(L). One
     measured number turns the open question into an answer.

WHY NOT SIMPLY RE-RUN IN FLOAT64. `precision.py` accepts WARP_DEM_PRECISION,
and it does switch every array and annotation. It does NOT produce a working
float64 solver, because Warp pins bare float literals in kernel bodies to
float32 and then refuses to mix them with float64 operands. Every literal in
every kernel would need an explicit cast. That is a real limitation of the
"flipping precision is a one-line change" claim in precision.py, it is recorded
in docs/precision.md, and it is work for Phase 7 rather than for Block 17.

    python scripts/drift_probe.py
"""

import numpy as np

from warp_dem import describe_device, resolve_device
from warp_dem.diagnostics import overlap_stats, position_drift_bound
from warp_dem.forces import ContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.precision import EPS, PRECISION, accumulation_bound
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import compute_budget
from warp_dem.walls import BoxBoundary

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

#: docs/precision.md, "Where it could bite": drum geometry, |x| <= 0.15 m,
#: 10 s of physical time at dt ~ 1e-5 s.
DRUM_COORD = 0.15
DRUM_STEPS = 1_000_000
CONTACT_OVERLAP = 20e-6


def ballistic_leg(device) -> None:
    """Constant velocity, no forces: the maximally correlated case."""
    print("\n=== 1. Ballistic drift (correlated increments) ===")
    print("Zero gravity, constant velocity. Every position increment is")
    print("identical, so roundings repeat in the same direction.\n")

    dt = 1e-5
    velocity = np.array([1.0, -2.0, 0.5])
    start = np.array([0.1, -0.2, 0.05])

    print(f"{'steps':>9}{'|error| [m]':>15}{'bound [m]':>13}{'err/step':>13}{'growth':>9}")
    previous = None
    for steps in (2_500, 5_000, 10_000, 20_000, 40_000):
        state = ParticleState.allocate(1, device, GLASS.radius, GLASS.density)
        state.set_positions([start.tolist()])
        state.set_velocities([velocity.tolist()])
        Solver(state, dt=dt, gravity=(0.0, 0.0, 0.0)).run(steps)

        simulated = np.asarray(state.positions()[0], dtype=np.float64)
        exact = start + velocity * (dt * steps)
        error = float(np.abs(simulated - exact).max())
        bound = accumulation_bound(float(np.abs(exact).max()), steps)
        growth = "" if previous is None else f"{error / previous:8.2f}x"
        previous = error
        print(f"{steps:9d}{error:15.4e}{bound:13.4e}{error / steps:13.4e}{growth:>9}")

    print("\nDoubling the steps roughly doubles the error, so growth is LINEAR")
    print("in N rather than sqrt(N). This is the correlated regime, and it is")
    print("the case docs/precision.md calls the worst case.")


def _velocity_sign_runs(series: np.ndarray) -> float:
    """Mean number of consecutive samples a component keeps its sign.

    series has shape (samples, particles, 3). Returns the mean run length in
    SAMPLES; the caller converts to steps.
    """
    signs = np.sign(series)
    signs[signs == 0] = 1
    changes = np.count_nonzero(np.diff(signs, axis=0), axis=0)
    # A component with `c` sign changes over `s` samples has s/(c+1) mean run.
    return float(np.mean(series.shape[0] / (changes + 1.0)))


def bed_leg(device):
    """A bed poured and settling: how correlated are the increments actually?

    SCOPE. This is a settling transient, not a tumbling drum. A bed that has
    come fully to rest has v ~ 0 and therefore accumulates NO drift at all —
    drift only happens while a particle is moving — so the transient, where
    particles fall, collide and rebound, is the drift-prone phase available at
    Block 17. A genuine drum measurement needs Phase 5 geometry and is flagged
    there.
    """
    print("\n=== 2. Correlation length in a poured, settling bed ===")
    print("A bed is the case that matters. If velocities reverse often, the")
    print("roundings decorrelate and the linear growth above does not apply.\n")

    n = 128
    r = GLASS.radius
    budget = compute_budget(
        r, GLASS.density, GLASS.youngs_modulus, GLASS.poisson_ratio,
        impact_velocity=1.0, hertz_steps=25,
    )
    dt = budget.limit

    # A 4 x 4 column, 8 deep, dropped a couple of radii onto the floor. Narrow
    # footprint so it settles into a genuine multi-layer bed with a force chain
    # rather than a single scattered layer.
    rng = np.random.default_rng(0)
    spacing = 2.2 * r
    grid = np.array(
        [(i, j, k) for k in range(8) for i in range(4) for j in range(4)],
        dtype=np.float64,
    )[:n] * spacing
    grid[:, 2] += 3.0 * r
    grid[:, :2] += 2.0 * r
    grid += rng.uniform(-0.05 * r, 0.05 * r, size=grid.shape)

    state = ParticleState.allocate(n, device, r, GLASS.density)
    state.set_positions(grid)

    # The box is SIZED FROM THE GRID, with a full radius of clearance on every
    # face. A first version hard-coded 20 mm while the grid reached 23 mm, so
    # the outermost column started buried 3 mm inside a wall — a 5 mm overlap
    # against a 2 mm radius. The bed left at 12 m/s and the measurement was of
    # an explosion rather than of settling. Never hard-code a domain that has
    # to contain generated geometry.
    lo = (0.0, 0.0, 0.0)
    hi = (
        float(grid[:, 0].max()) + 2.0 * r,
        float(grid[:, 1].max()) + 2.0 * r,
        float(grid[:, 2].max()) + 10.0 * r,
    )

    model = ContactModel(state, GLASS.pair_params())
    boundary = BoxBoundary(state, GLASS.wall_params(), lo, hi)
    solver = Solver(state, dt=dt, gravity=(0.0, 0.0, -9.81), budget=budget,
                    contact_model=model, boundary=boundary)

    samples = 300
    every = 100
    series = np.empty((samples, n, 3))
    speed = np.empty(samples)
    for s in range(samples):
        solver.run(every)
        series[s] = state.velocities()
        speed[s] = float(np.linalg.norm(series[s], axis=1).mean())

    run_samples = _velocity_sign_runs(series)
    correlation_steps = run_samples * every

    stats = overlap_stats(state, model.contacts, boundary)
    coord = float(np.abs(state.positions()).max())

    print(f"particles                  : {n}")
    print(f"steps run                  : {solver.step_count:,}")
    print(f"timestep                   : {dt * 1e6:.3f} us")
    print(f"largest |coordinate|       : {coord:.4f} m")
    print(f"mean speed, first / last   : {speed[0]:.4e} / {speed[-1]:.4e} m/s")
    print(f"overlap                    : {stats.describe()}")
    print()
    print(f"mean velocity sign-run     : {correlation_steps:.1f} steps "
          f"({correlation_steps / solver.step_count:.1%} of the run)")
    print(f"drift enhancement over the decorrelated estimate: "
          f"sqrt(L) = {np.sqrt(correlation_steps):.1f}x")
    print(f"  (fully correlated would be sqrt(N) = "
          f"{np.sqrt(solver.step_count):.0f}x)")

    return correlation_steps, stats


def extrapolate(correlation_steps: float) -> None:
    """Apply the measured correlation length to the drum case in §5.1."""
    print("\n=== 3. Extrapolation to drum geometry ===")
    print(f"|x| <= {DRUM_COORD} m, {DRUM_STEPS:,} steps, "
          f"contact overlap ~{CONTACT_OVERLAP * 1e6:.0f} um\n")

    per_add = 0.5 * EPS * DRUM_COORD
    correlated = per_add * DRUM_STEPS
    decorrelated = per_add * np.sqrt(DRUM_STEPS)
    measured = per_add * np.sqrt(max(correlation_steps, 1.0) * DRUM_STEPS)

    rows = (
        ("fully correlated (worst case)", correlated),
        ("measured correlation length", measured),
        ("fully decorrelated", decorrelated),
    )
    print(f"{'case':>32}{'drift':>14}{'vs overlap':>13}")
    for label, value in rows:
        print(f"{label:>32}{value * 1e6:11.2f} um{value / CONTACT_OVERLAP:12.3f}x")

    print()
    if measured < 0.1 * CONTACT_OVERLAP:
        print("VERDICT: drift is at least an order of magnitude below the contact")
        print("overlap, so float32 positions are safe for the drum runs. The")
        print("open question in docs/precision.md section 5.1 is now answered")
        print("with a measurement rather than an argument.")
    else:
        print("VERDICT: drift is NOT comfortably below the contact overlap.")
        print("Positions should move to float64, or be stored relative to a")
        print("local origin, before Phase 5.")


def main() -> None:
    device = resolve_device("auto")
    print(describe_device(device))
    print(f"working precision: {PRECISION}  (EPS = {EPS:.3e})")

    ballistic_leg(device)
    correlation_steps, _ = bed_leg(device)
    extrapolate(correlation_steps)

    state = ParticleState.allocate(1, device, GLASS.radius, GLASS.density)
    state.set_positions([[DRUM_COORD, 0.0, 0.0]])
    print(f"\nrigorous bound helper: position_drift_bound(state, {DRUM_STEPS}) = "
          f"{position_drift_bound(state, DRUM_STEPS):.4e} m")


if __name__ == "__main__":
    main()
