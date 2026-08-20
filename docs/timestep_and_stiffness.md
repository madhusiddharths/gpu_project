# Timestep bounds and stiffness scaling

**Computed:** Block 8. Reference material: soda-lime glass beads,
R = 2 mm, rho = 2500 kg/m3, nu = 0.20.

## Two bounds

| | formula | scaling |
|---|---|---|
| Rayleigh wave | `t_R = pi R sqrt(rho/G) / (0.1631 nu + 0.8766)` | `R`, `rho^1/2`, `E^-1/2` |
| Hertz contact | `t_c = 2.87 (m_eff^2 / (R_eff E_eff^2 v))^(1/5)` | `R`, `rho^2/5`, `E^-2/5`, `v^-1/5` |

Convention used: `dt <= min(0.2 t_R, t_c / 25)`.

> WARNING: the 0.1631/0.8766 fit and the 2.87 prefactor are quoted throughout
> the DEM literature but are not yet confirmed here against primary sources.
> Same status as Beverloo's constants in `validation_targets.md`. Confirm
> before Phase 3 reports against them.

## Measured cost of stiffness

| E [Pa] | softening | 0.2 t_R [us] | t_c/25 [us] | dt limit [us] | binds | steps / 10 s |
|---|---|---|---|---|---|---|
| 6.8e10 (real) | 1x | 0.41 | 0.49 | **0.41** | Rayleigh | 24,357,793 |
| 1.0e9 | 68x | 3.39 | 2.64 | **2.64** | Hertz | 3,787,878 |
| 1.0e8 (**in use**) | 680x | 10.71 | 6.64 | **6.64** | Hertz | 1,506,024 |
| 1.0e7 | 6,800x | 33.85 | 16.68 | **16.68** | Hertz | 599,655 |
| 5.0e6 (former) | 13,600x | 47.88 | 22.00 | **22.00** | Hertz | 454,454 |

Impact velocity 1 m/s throughout. Note that the binding constraint flips from
Rayleigh to Hertz between 1e10 and 1e9 Pa, so the remedy for a violation is
stiffness-dependent.

## Static overlap, delta/R [%]

Load expressed as N particle weights — the depth of bed above the contact.

| E [Pa] | N=1 | N=10 | N=20 | N=50 |
|---|---|---|---|---|
| 6.8e10 | 0.000 | 0.002 | 0.002 | 0.005 |
| 1.0e8 | 0.026 | 0.121 | 0.191 | 0.352 |
| 1.0e7 | 0.121 | 0.559 | 0.888 | 1.636 |
| 5.0e6 | 0.191 | 0.888 | **1.410** | **2.597** |

## The decision

`glass_beads.youngs_modulus` moved from 5.0e6 to **1.0e8 Pa** in Block 8.

Rationale: 5.0e6 violates Validation Target 5 (delta/R < 1%) for beds deeper
than roughly 15 particles. An angle-of-repose pile is 30-50 particles tall, and
a measurably soft bed slumps flatter — the documented #1 cause of failed repose
validation. 1.0e8 keeps overlap at 0.35% for a 50-deep load while still
retaining a 16x cost reduction against real glass.

The criterion applied here (delta/R < 1%) was pre-registered in
`validation_targets.md` before any solver code existed. That is what
distinguishes this from tuning a parameter until the answer looks right.

## What still has to be earned

This is a JUSTIFIED choice, not a VALIDATED one. Validation Target 5 requires
demonstrating that the reported quantities — repose angle, discharge rate, flow
regime — are insensitive to the scaling across the range used. Planned for
Phase 3: repeat the repose measurement at 1e8, 3e8 and 1e9 Pa and show the
angle changes by less than the experimental uncertainty (+/- 0.8 deg, Sunday et
al. 2020). If it moves, the scaling is too aggressive and the timestep budget
has to be paid.