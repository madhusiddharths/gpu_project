"""Run the Phase 1 solver end to end from a YAML config.

    python scripts/run.py
    python scripts/run.py material=glass_beads run.name=repose run.steps=20000
    python scripts/run.py material=tablet_placebo run.n_particles=1024
    python scripts/run.py device=cpu run.dt=1e-6

Hydra writes each run into its own timestamped directory under outputs/, and
this script drops a `summary.txt` there alongside the resolved config, so a
result is never separated from the inputs that produced it.

WHAT THIS SCRIPT IS FOR. It is the Phase 1 done-when: proof that the solver is
driven entirely by configuration rather than by edited constants. Every number
that matters — material, geometry, particle count, timestep — comes from YAML,
and the device is resolved in exactly one place. Phase 3's validation cases are
this script with different configs.
"""

from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from warp_dem import describe_device, resolve_device
from warp_dem.diagnostics import energy_breakdown, overlap_stats, position_drift_bound
from warp_dem.forces import ContactModel
from warp_dem.materials import MaterialProperties
from warp_dem.packing import lattice_fill
from warp_dem.solver import Solver
from warp_dem.state import ParticleState
from warp_dem.timestep import assert_timestep_valid, compute_budget
from warp_dem.walls import BoxBoundary


def build(cfg: DictConfig):
    """Assemble state, solver and diagnostics from a resolved config.

    Returns (solver, model, boundary, material, budget). Separated from `main`
    so tests can drive it through Hydra's compose API without a subprocess.
    """
    device = resolve_device(cfg.device)
    material = MaterialProperties.from_config(cfg.material)

    budget = compute_budget(
        material.radius,
        material.density,
        material.youngs_modulus,
        material.poisson_ratio,
        impact_velocity=float(cfg.run.impact_velocity),
        hertz_steps=int(cfg.run.hertz_steps),
    )
    # A null dt takes the derived bound. An explicit one is still checked
    # against it, so an unstable hand-set timestep fails loudly rather than
    # producing a plausible-looking bed that is quietly wrong.
    dt = budget.limit if cfg.run.dt is None else float(cfg.run.dt)
    assert_timestep_valid(dt, budget)

    positions = lattice_fill(
        int(cfg.run.n_particles),
        material.radius,
        cfg.geometry.bounds_min,
        cfg.geometry.bounds_max,
        spacing_factor=float(cfg.run.spacing_factor),
        jitter=float(cfg.run.jitter),
        seed=int(cfg.seed),
        drop_height=float(cfg.run.drop_height),
    )

    state = ParticleState.allocate(
        len(positions), device, material.radius, material.density
    )
    state.set_positions(positions)

    model = ContactModel(
        state,
        material.pair_params(),
        slots_per_particle=int(cfg.contacts.slots_per_particle),
        check_overflow=bool(cfg.contacts.check_overflow),
    )
    boundary = BoxBoundary.from_config(state, material.wall_params(), cfg.geometry)
    solver = Solver(
        state,
        dt=dt,
        gravity=tuple(float(g) for g in cfg.geometry.gravity),
        budget=budget,
        contact_model=model,
        boundary=boundary,
    )
    return solver, model, boundary, material, budget


def report(solver, model, boundary, material, lines) -> None:
    """One diagnostic line, echoed to stdout and captured for the summary."""
    state = solver.state
    stats = overlap_stats(state, model.contacts, boundary)
    energy = energy_breakdown(
        state, solver.gravity, model.contacts, material.pair_params().e_eff
    )
    speed = float(np.linalg.norm(state.velocities(), axis=1).mean())
    line = (
        f"{solver.time * 1e3:9.3f} ms  step {solver.step_count:8d}  "
        f"contacts {stats.contacts:6d}  mean d/R {stats.mean:7.4%}  "
        f"max d/R {stats.maximum:7.4%}  <v> {speed:9.3e} m/s  "
        f"E {energy.total:11.4e} J"
    )
    print(line)
    lines.append(line)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    solver, model, boundary, material, budget = build(cfg)
    state = solver.state

    header = [
        "--- Resolved config ---",
        OmegaConf.to_yaml(cfg),
        "--- Device ---",
        describe_device(resolve_device(cfg.device)),
        "--- Timestep budget ---",
        budget.describe(),
        f"timestep in use   : {solver.dt:.3e} s"
        f"{'  (derived)' if cfg.run.dt is None else '  (from config)'}",
        f"particles         : {state.n}",
        f"material          : {material.name}",
        "",
    ]
    for line in header:
        print(line)

    lines = []
    report(solver, model, boundary, material, lines)
    remaining = int(cfg.run.steps)
    every = max(1, int(cfg.run.output_every))
    while remaining > 0:
        chunk = min(every, remaining)
        solver.run(chunk)
        remaining -= chunk
        report(solver, model, boundary, material, lines)

    stats = overlap_stats(state, model.contacts, boundary)
    footer = [
        "",
        "--- Validation Target 5 (overlap) ---",
        stats.describe(),
        "",
        f"float32 position drift bound over {solver.step_count} steps: "
        f"{position_drift_bound(state, solver.step_count):.3e} m",
        "  (a BOUND, not a measurement -- see scripts/drift_probe.py and",
        "   docs/precision.md for the measured correlation length)",
        f"\nTarget 5: {stats.verdict}",
    ]
    for line in footer:
        print(line)

    # Write to Hydra's OWN output directory, asked for explicitly.
    #
    # Do NOT assume the process has been chdir'd there. Hydra 1.1 changed the
    # default of `hydra.job.chdir` to False, and with `version_base=None` that
    # is what applies — so a bare open("summary.txt") lands in whatever
    # directory the command was launched from and every run silently overwrites
    # the last. Which is precisely the "a result is never separated from the
    # inputs that produced it" property this file claims to provide.
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.txt"
    summary.write_text("\n".join([*header, *lines, *footer]) + "\n")
    print(f"\nsummary written to {summary}")


if __name__ == "__main__":
    main()
