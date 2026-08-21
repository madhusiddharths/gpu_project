"""Hydra wiring and initial packing — Block 18.

Phase 1's done-when is that the whole solver runs from a YAML. This file
asserts that claim rather than trusting the smoke run: every number that
matters — material, geometry, particle count, timestep, device — comes from
config, and nothing is an edited constant.

The config is built with Hydra's COMPOSE API rather than by launching a
subprocess. Composing exercises the same defaults list, the same overrides and
the same schema, but keeps the assertion inside the test process where a
failure produces a stack trace instead of an exit code.

`scripts/run.py` is loaded by path because `scripts/` is not an installed
package. That is deliberate — the scripts are entry points, not library code,
and making them importable would invite the solver to grow a dependency on one.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from hydra import compose, initialize_config_dir

from warp_dem import resolve_device
from warp_dem.diagnostics import overlap_stats
from warp_dem.packing import lattice_fill

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
RADIUS = 0.002


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "warp_dem_run_script", REPO / "scripts" / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN = _load_run_module()


def _config(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def _available_devices():
    devices = ["cpu"]
    if wp.get_cuda_devices():
        devices.append("cuda:0")
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device_name(request):
    return request.param


# ── packing ───────────────────────────────────────────────────────────────────


def test_packing_keeps_every_particle_clear_of_the_walls():
    """The postcondition the module exists for.

    Block 17 hit the opposite twice: a lattice generated first and a domain
    hard-coded second, with the lattice overrunning the domain. A 5 mm overlap
    on a 2 mm particle launched the bed at 12 m/s and nothing crashed.
    """
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.05, 0.05, 0.05])
    positions = lattice_fill(200, RADIUS, lo, hi)

    assert positions.shape == (200, 3)
    assert np.all(positions >= lo + RADIUS)
    assert np.all(positions <= hi - RADIUS)


def test_packing_particles_do_not_overlap_each_other():
    positions = lattice_fill(150, RADIUS, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    assert distance.min() > 2.0 * RADIUS, (
        f"closest pair is {distance.min():.4e} m apart, under {2 * RADIUS:.4e}"
    )


def test_packing_refuses_to_overfill_and_says_what_would_fit():
    """An error message that names the capacity beats one that says 'too many'."""
    with pytest.raises(ValueError, match="capacity is"):
        lattice_fill(100_000, RADIUS, (0.0, 0.0, 0.0), (0.02, 0.02, 0.02))


def test_packing_rejects_a_spacing_that_jitter_would_close():
    with pytest.raises(ValueError, match="too tight for jitter"):
        lattice_fill(8, RADIUS, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05),
                     spacing_factor=2.05, jitter=0.05)


def test_packing_rejects_a_domain_smaller_than_one_particle():
    with pytest.raises(ValueError, match="too small"):
        lattice_fill(1, RADIUS, (0.0, 0.0, 0.0), (0.001, 0.001, 0.001))


def test_packing_is_reproducible_from_the_seed():
    a = lattice_fill(50, RADIUS, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05), seed=7)
    b = lattice_fill(50, RADIUS, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05), seed=7)
    c = lattice_fill(50, RADIUS, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05), seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ── config wiring ─────────────────────────────────────────────────────────────


def test_default_config_composes_and_builds(device_name):
    cfg = _config(f"device={device_name}", "run.n_particles=32")
    solver, model, boundary, material, budget = RUN.build(cfg)

    assert solver.state.n == 32
    assert material.name == "glass_beads"
    assert solver.state.device == resolve_device(device_name)
    assert budget.binding in ("Rayleigh", "Hertz")


def test_null_dt_is_derived_from_the_timestep_budget(device_name):
    cfg = _config(f"device={device_name}", "run.n_particles=16")
    assert cfg.run.dt is None
    solver, _, _, _, budget = RUN.build(cfg)
    assert solver.dt == pytest.approx(budget.limit)


def test_an_explicit_dt_is_still_checked_against_the_bound(device_name):
    """A hand-set timestep that violates stability must fail loudly.

    An unstable DEM run does not crash — it produces a plausible-looking bed
    that is quietly wrong, which is far worse than an exception.
    """
    from warp_dem.timestep import TimestepError

    cfg = _config(f"device={device_name}", "run.n_particles=16", "run.dt=1e-3")
    with pytest.raises(TimestepError, match="exceeds the stability limit"):
        RUN.build(cfg)


def test_an_explicit_valid_dt_is_honoured(device_name):
    cfg = _config(f"device={device_name}", "run.n_particles=16", "run.dt=1e-6")
    solver, _, _, _, _ = RUN.build(cfg)
    assert solver.dt == pytest.approx(1e-6)


def test_material_selection_reaches_the_solver(device_name):
    cfg = _config(f"device={device_name}", "material=tablet_placebo",
                  "run.n_particles=16")
    _, _, _, material, _ = RUN.build(cfg)
    assert material.name == "tablet_placebo"
    assert material.density == pytest.approx(1250.0)
    # tablet_placebo has no restitution_wall key, so it must default.
    assert material.restitution_wall == material.restitution


def test_geometry_selection_reaches_the_boundary(device_name):
    cfg = _config(f"device={device_name}", "run.n_particles=16")
    _, _, boundary, _, _ = RUN.build(cfg)
    assert tuple(boundary.bounds_max) == pytest.approx(
        tuple(float(v) for v in cfg.geometry.bounds_max)
    )


def test_contact_slot_count_reaches_the_history(device_name):
    cfg = _config(f"device={device_name}", "run.n_particles=16",
                  "contacts.slots_per_particle=8")
    _, model, _, _, _ = RUN.build(cfg)
    assert model.history.slots_per_particle == 8


def test_overfilling_the_domain_fails_with_an_actionable_message(device_name):
    cfg = _config(f"device={device_name}", "run.n_particles=100000")
    with pytest.raises(ValueError, match="run.n_particles"):
        RUN.build(cfg)


# ── end to end ────────────────────────────────────────────────────────────────


def test_a_short_run_settles_into_a_bed_meeting_target_5(device_name):
    """Phase 1's done-when, in miniature.

    Particles are poured from config, fall, collide, and settle. Asserting that
    the bed FORMS matters as much as the overlap figure: with no contacts the
    Target 5 inequality holds trivially, which is how a run that never formed a
    bed gets mistaken for a good one.
    """
    cfg = _config(
        f"device={device_name}",
        "run.n_particles=200",
        "geometry.bounds_max=[0.022,0.022,0.12]",
    )
    solver, model, boundary, material, _ = RUN.build(cfg)
    solver.run(20_000)

    stats = overlap_stats(solver.state, model.contacts, boundary)
    assert stats.contacts > 100, f"bed did not form: {stats.describe()}"
    assert stats.within_target, stats.describe()
    assert stats.verdict == "PASS"

    positions = solver.state.positions()
    lo = np.array([float(v) for v in cfg.geometry.bounds_min])
    hi = np.array([float(v) for v in cfg.geometry.bounds_max])
    assert np.all(positions > lo - material.radius), "a particle escaped the floor"
    assert np.all(positions < hi + material.radius), "a particle escaped the ceiling"

    speed = float(np.linalg.norm(solver.state.velocities(), axis=1).mean())
    assert speed < 0.05, f"bed had not begun to settle: <v> = {speed:.3e} m/s"


def test_energy_only_decreases_in_a_dissipative_run(device_name):
    """Glass beads have e = 0.90 and friction, so the total must fall.

    A rising total would mean a contact was injecting energy — the signature of
    a tangential spring that is not being truncated on Coulomb clipping.
    """
    cfg = _config(
        f"device={device_name}",
        "run.n_particles=120",
        "geometry.bounds_max=[0.022,0.022,0.12]",
    )
    solver, model, boundary, material, _ = RUN.build(cfg)

    lines = []
    RUN.report(solver, model, boundary, material, lines)
    from warp_dem.diagnostics import energy_breakdown

    e_eff = material.pair_params().e_eff
    series = []
    for _ in range(10):
        solver.run(1500)
        series.append(
            energy_breakdown(solver.state, solver.gravity, model.contacts, e_eff).total
        )

    assert series[-1] < series[0], "a dissipative run did not lose energy"
    rises = [b - a for a, b in zip(series, series[1:], strict=False) if b > a]
    assert not rises or max(rises) < 0.02 * abs(series[0]), (
        f"energy rose by {max(rises):.3e} J during a dissipative run"
    )
    assert lines and "contacts" in lines[0]


def test_summary_is_written_into_the_hydra_run_directory(tmp_path):
    """The script must not assume Hydra has chdir'd it into the output dir.

    Hydra 1.1 changed the default of `hydra.job.chdir` to False, and
    `version_base=None` selects that behaviour — so a bare
    `open("summary.txt", "w")` lands in whatever directory the command was
    launched from, and every run silently overwrites the last. That is exactly
    the property run.py claims to provide ("a result is never separated from
    the inputs that produced it"), so it gets an end-to-end test rather than a
    comment.

    Run as a SUBPROCESS deliberately. The bug lives in the interaction between
    Hydra's launcher and the process working directory, which the compose API
    used by the rest of this file does not exercise at all.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "run.py"),
            "device=cpu",
            "run.name=pytest_summary",
            "run.steps=20",
            "run.output_every=20",
            "run.n_particles=8",
            f"hydra.run.dir={tmp_path}/${{run.name}}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    run_dir = tmp_path / "pytest_summary"
    summary = run_dir / "summary.txt"
    assert summary.exists(), f"summary.txt not in the run dir; got {list(tmp_path.iterdir())}"
    assert not (tmp_path / "summary.txt").exists(), "summary leaked into the launch dir"

    text = summary.read_text()
    assert "Resolved config" in text
    assert "Target 5" in text
    # The resolved config must sit beside it, or the summary is orphaned.
    assert (run_dir / ".hydra" / "config.yaml").exists()


def test_run_directory_is_named_after_the_run():
    """Two runs must be distinguishable in a listing.

    The Block 18 rewrite of config.yaml dropped the `hydra.run.dir` block, so
    output directories became bare timestamps. Cheap to lose, annoying to
    notice later when a sweep has produced thirty identical-looking folders.
    """
    # Read the file as TEXT. Hydra strips the `hydra:` node out of the composed
    # config (it lives in HydraConfig), and OmegaConf.load resolves the
    # interpolations on access — so neither route can see the pattern itself.
    # The file is what regressed, so the file is what gets asserted.
    raw = (CONFIGS / "config.yaml").read_text()
    assert "hydra:" in raw, "the hydra: block is missing from config.yaml"
    assert "${run.name}" in raw, "run directories would not carry the run name"
    assert "${now:" in raw, "run directories would not be timestamped"
