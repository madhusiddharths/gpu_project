"""Print the timestep budget for the configured material.

    python scripts/timestep_report.py
    python scripts/timestep_report.py material=tablet_placebo
    python scripts/timestep_report.py material=glass_beads impact_velocity=3.0
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from warp_dem.timestep import compute_budget, hertz_static_overlap, sphere_mass

# Material configs have gone through one restructure already; look in both the
# nested contact block and the flat legacy location rather than assuming.
_PATHS = {
    "youngs_modulus": ["material.contact.youngs_modulus", "material.youngs_modulus"],
    "poisson_ratio": ["material.contact.poisson_ratio", "material.poisson_ratio"],
    "radius": ["material.radius", "material.sphere_equivalent_radius"],
    "density": ["material.density"],
}


def pick(cfg: DictConfig, key: str) -> float:
    for path in _PATHS[key]:
        value = OmegaConf.select(cfg, path)
        if value is not None:
            return float(value)
    raise KeyError(
        f"could not find {key!r} in the material config; tried {_PATHS[key]}.\n"
        f"Material keys present: {sorted(cfg.material.keys())}"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    radius = pick(cfg, "radius")
    density = pick(cfg, "density")
    modulus = pick(cfg, "youngs_modulus")
    poisson = pick(cfg, "poisson_ratio")
    v_impact = float(OmegaConf.select(cfg, "impact_velocity") or 1.0)

    budget = compute_budget(radius, density, modulus, poisson, impact_velocity=v_impact)

    print(f"\nmaterial: {cfg.material.name}")
    print(f"  radius {radius * 1e3:.3f} mm   density {density:.0f} kg/m3   "
          f"E {modulus:.3e} Pa   nu {poisson:.3f}\n")
    print(budget.describe())

    print(f"\nsteps required per second of physical time: {int(1.0 / budget.limit):,}")

    print("\nstatic overlap under a stack of N particle weights "
          "(Validation Target 5: delta/R < 1%)")
    weight = sphere_mass(radius, density) * 9.81
    print(f"  {'N':>5} {'delta [um]':>12} {'delta/R [%]':>12}  status")
    for n in (1, 10, 20, 50, 100):
        delta = hertz_static_overlap(n * weight, radius, modulus, poisson)
        ratio = delta / radius * 100.0
        print(f"  {n:5d} {delta * 1e6:12.3f} {ratio:12.3f}  "
              f"{'ok' if ratio < 1.0 else 'TOO SOFT'}")
    print()


if __name__ == "__main__":
    main()