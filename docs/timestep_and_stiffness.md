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
---

## Steps per collision — measured, Block 10

`compute_budget(..., hertz_steps=N)` demands N timesteps inside one collision.
25 was a convention carried from the literature. Measured relative error in
coefficient of restitution, head-on two-sphere collisions at 1 m/s, glass beads
at E = 1e8 Pa:

| e_in | spc=10 | spc=15 | spc=25 | spc=50 | spc=100 |
|---|---|---|---|---|---|
| 0.50 | 0.933% | 0.933% | 1.323% | 0.084% | 0.163% |
| 0.70 | 1.729% | 1.729% | 0.771% | 0.323% | 0.144% |
| 0.90 | 0.411% | 0.411% | 0.080% | 0.057% | 0.058% |
| 0.97 | 0.162% | 0.162% | 0.043% | 0.011% | 0.003% |

**spc=10 and spc=15 are identical because the Rayleigh bound clamps dt.** At
E = 1e8 Pa the Rayleigh limit is 10.71 us and t_c/10 would be 16.60 us, so a
request for 10 steps per collision silently delivers 15. `budget.limit` — the
min of the two bounds — is the only correct thing to hand a Solver;
`budget.hertz_limit` alone is a bug, and `assert_timestep_valid` caught exactly
that mistake when this script was first written.

### The error is contact PHASE, not float32

Shifting the initial gap by a fraction of one `v*dt` step moves where in the
timestep contact begins:

| e | spc | spread in measured e across phase |
|---|---|---|
| 0.50 | 25 | 0.0118 |
| 0.50 | 50 | 0.0040 |
| 0.70 | 25 | 0.0091 |
| 0.90 | 25 | 0.0007 |

The spread exceeds the error at any single phase, so phase — not arithmetic — is
the dominant term. A collision begins mid-step, so the first overlap is quantised
by `v*dt`; fewer steps per collision misresolves a larger fraction of the entry
and exit, and low `e` amplifies it because damping dominates the force there.

### Decision

**`hertz_steps` stays at 25.** Worst case 1.32% against a 2% target is a 1.5x
margin, but that worst case is e = 0.50, which this project never simulates. The
materials actually used sit far from it: glass beads (e = 0.90) give **0.08%**,
tablet placebo (e = 0.60) roughly 1%. Moving to 50 would double the step count
for every run in the project to buy margin at a restitution never used.

Revisit if a future material has e below 0.6, or if Phase 3 validation shows
sensitivity to it.
