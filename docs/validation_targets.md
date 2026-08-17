# Validation targets

**Committed:** 2026-08-16
**Status:** PRE-REGISTERED — written before any solver code existed.

These targets were fixed in advance. If a validation fails, the response is to debug
the physics — not to adjust the target or tune parameters until it passes. Any change
to this file after the commit date must be justified in the git commit message and
disclosed in the writeup.

**Debug order when a validation fails:**

1. Tangential history not carried correctly between timesteps
2. Timestep violating the Rayleigh criterion
3. Rolling resistance missing or mis-scaled
4. Wall friction unset or set on the wrong surface

---

## 0. Source list

| Key | Citation | Used for |
|---|---|---|
| **[Sunday2020]** | Sunday, Murdoch, Tardivel, Schwartz & Michel (2020). *Validating N-body code Chrono for granular DEM simulations in reduced-gravity environments.* MNRAS 498(1), 1062–1079. arXiv:2009.10448 | Primary reference. Repose angle, glass-bead material properties, drum flow regimes, dynamic repose vs Froude. **Open access.** |
| **[Zhou2002]** | Zhou, Xu, Yu & Zulli (2002). *An experimental and numerical study of the angle of repose of coarse spheres.* Powder Technology 125(1), 45–54. doi:10.1016/S0032-5910(01)00520-4 | Repose-angle parameter sensitivity; rolling friction baseline |
| **[Beverloo1961]** | Beverloo, Leniger & van de Velde (1961). *The flow of granular solids through orifices.* Chemical Engineering Science 15(3–4), 260–269. | Hopper discharge correlation |
| **[Mellmann2001]** | Mellmann (2001). *The transverse motion of solids in rotating cylinders — forms of motion and transition behaviour.* Powder Technology 118(3), 251–270. doi:10.1016/S0032-5910(00)00402-2 | Flow-regime classification and the Bed Behaviour Diagram |
| **[Just2013]** | Just, Toschkoff, Funke, Djuric, Scharrer, Khinast, Knop & Kleinebudde (2013). *Experimental analysis of tablet properties for discrete element modeling of an active coating process.* AAPS PharmSciTech 14(1), 402–411. doi:10.1208/s12249-013-9925-5 | Tablet material properties, dynamic repose in a drum coater. **Open access (PMC3581635).** |
| **[Suzzi2012]** | Suzzi, Toschkoff, Radl, Machold, Fraser, Glasser & Khinast (2012). *DEM simulation of continuous tablet coating: effects of tablet shape and fill level on inter-tablet coating variability.* Chemical Engineering Science 69(1), 107–121. | Glued-sphere tablet shapes; CV framing |
| **[Toschkoff2013]** | Toschkoff & Khinast (2013). *Mathematical modeling of the coating process.* International Journal of Pharmaceutics 457(2), 407–422. | Review; scale-limitation framing for section 1.1 |
| **[Boehling2016]** | Boehling, Toschkoff, Knop, Kleinebudde, Just, Funke, Rehbaum & Khinast (2016). *Analysis of large-scale tablet coating: modeling, simulation and experiments.* European Journal of Pharmaceutical Sciences 90, 14–24. | Industrial-scale GPU DEM reference point (~1.03 M tablets, 290 kg) |
| **[Foerster1994]** | Foerster, Louge, Chang & Allia (1994). Physics of Fluids 6(3), 1108–1115. | Glass-bead particle–particle restitution |
| **[Alizadeh2014]** | Alizadeh, Bertrand & Chaouki (2014). AIChE Journal 60(1), 60–75. | Glass-bead wall restitution and friction |

**Verification status.** Values below marked ✅ were read from the full text of an
open-access source. Values marked ⚠️ come from secondary reporting (review articles,
abstracts, citing papers) and MUST be confirmed against the primary PDF before Phase 3
begins. Do not run a validation against a ⚠️ number.

---

## Target 1 — Angle of repose (static pile)

**Material:** soda-lime glass beads, 1 mm diameter
**Primary source:** [Sunday2020] §5, piling experiment

| Quantity | Value | Status |
|---|---|---|
| Experimental repose angle | **25.2 ± 0.8°** (mean ± s.d. of the mean, 6 trials / 12 measurements) | ✅ |
| Reference simulated angle | 25.3 ± 0.1° at μ_r = 0.09 | ✅ |
| Calibrated rolling friction | μ_r = 0.09 (swept 0 → 0.2) | ✅ |
| Particle count in reference sim | 58,040 | ✅ |

**PASS: simulated repose angle in 22.2° – 28.2°** (target ± 3°).

A ± 3° window is not slack — it is the honest experimental scatter. [Zhou2002] reports a
maximum estimation error of about 3° for their predictive formula, and repose angles for
glass beads in rotating drums have been measured across a wide band (roughly 22–39° for
lower angles) depending on drum size and method. A tighter tolerance would be claiming
precision the underlying measurement does not have.

### Measurement protocol (fixed now, not adjustable later)

- Particle count: 40,000 (M3 Pro CPU backend; comfortable overnight)
- Particle diameter: 1.0 mm ± 0.2 mm (normal distribution, as [Sunday2020])
- Container: rectangular, thickness ≥ 20 particle diameters. [Zhou2002] found repose
  angle depends on container thickness up to about 20 d, above which it is constant.
  Running thinner without saying so would silently inflate the angle.
- Wall condition: rough. [Sunday2020] fixed a layer of particles to the floor and ramps
  rather than using a smooth plane. Replicate this — a frictionless floor gives a
  visibly wrong pile.
- Settling criterion: total kinetic energy < 1 × 10⁻⁹ J, then hold 0.5 s
- Surface fit: linear regression through the upper-edge particles, **excluding the
  outer 15% (tail-end curvature) and inner 15% (funnel shadow)**. [Sunday2020] excludes
  tail-ends and centre for exactly this reason. Fixing the exclusion window now removes
  a tuning knob.
- Report left and right angles separately; the mean is the result, the difference is
  a sanity check on symmetry.

### Material properties — glass beads

All from [Sunday2020] Table 2 unless noted.

| Property | Symbol | Value | Status |
|---|---|---|---|
| Density | ρ | 2500 kg/m³ | ✅ |
| Diameter | d | 1.0 ± 0.2 mm | ✅ |
| Young's modulus (true) | E | ~70 GPa | ✅ |
| Young's modulus (simulation) | E* | 70 MPa | ✅ |
| Poisson ratio | ν | 0.24 | ✅ |
| Restitution, particle–particle | e_pp | 0.97 [Foerster1994] | ✅ |
| Restitution, particle–wall | e_pw | 0.82 [Alizadeh2014] | ✅ |
| Static friction, particle–particle | μ_s,pp | 0.16 [Alizadeh2014] | ✅ |
| Static friction, particle–wall | μ_s,pw | 0.45 [Alizadeh2014] | ✅ |
| Rolling friction | μ_r | 0.09 (calibrated against repose) | ✅ |
| Spinning friction | μ_t | 0 | ✅ |

**Note on μ_r.** This is a calibrated parameter, not a measured one. [Sunday2020] is
explicit that rolling and spinning friction coefficients are model-dependent and are
obtained by fitting simulations to experiment. Using 0.09 as an *input* and then
"validating" against the same repose angle it was fitted to would be circular. The
non-circular use is: adopt μ_r = 0.09 from the literature, then check that Targets 2 and
3 — hopper discharge and drum flow regimes — also pass without further adjustment.

---

## Target 2 — Hopper discharge vs Beverloo

**Primary source:** [Beverloo1961]

```
W = C · ρ_bulk · √g · (D_orifice − k · d_particle)^2.5
```

W = mass flow rate (kg/s), ρ_bulk = bulk density of the *flowing* material,
D_orifice = orifice diameter, d_particle = particle diameter.

| Constant | Value adopted | Reported range | Status |
|---|---|---|---|
| C (discharge coefficient) | **0.58** | 0.55–0.65 typical; 0.64 reported for glass spheres (Grace & Raffle 1986) | ⚠️ |
| k (shape coefficient) | **1.4** | ~1.5 for spherical particles; 1–2 general; 2.9 for sand in the original paper | ⚠️ |

Both constants are corroborated across several independent sources, but confirm the
values in the [Beverloo1961] primary text before running.

**PASS: simulated W within ± 10% of predicted, for at least 3 orifice diameters.**

### Validity conditions

Beverloo does not apply outside these. Record them per run; a "failure" outside the
valid range is not a solver failure.

- Flat-bottomed hopper, circular orifice, free discharge, no cohesion
- Coarse free-flowing material (> 400 µm) — 1 mm glass beads qualify
- **D_orifice / d_particle ≈ 10.** Recent DEM work (arXiv:2512.03698) found good
  agreement with Beverloo scaling at D/d = 10 with sufficient bed height, and a
  breakdown at D/d = 20, where discharge decays exponentially with time because the
  constant-pressure assumption fails. Stay near 10.
- **Bed height: h/D_orifice > 2, equivalently h/d > 20.** Same source proposes this as
  the dimensionless criterion for height-independent discharge. Below it, flow rate
  depends on fill height and the correlation does not hold.
- Hopper diameter / orifice diameter ≥ 2.5

### Test matrix

Particle d = 1.0 mm throughout. ρ_bulk measured from the settled bed in-simulation,
not assumed.

| D_orifice (mm) | D/d | Predicted W (kg/s) | Simulated W | Error % |
|---|---|---|---|---|
| 8 | 8 | | | |
| 10 | 10 | | | |
| 12 | 12 | | | |
| 14 | 14 | | | |

Measure W over the **steady window only** — after the initial transient and before the
bed drains below the h/D > 2 criterion. Fit a straight line to discharged mass vs time
and take the slope; do not divide total mass by total time.

---

## Target 3 — Rotating drum flow regimes

**Primary source:** [Sunday2020] §6, Table 3 and Fig. 12.
**Classification framework:** [Mellmann2001]

```
Fr = ω²R / g          (equivalently ω²D / 2g)
```

### Regime transitions

Observed in [Sunday2020] for 1 mm glass beads, 60 mm drum, 50% fill, μ_r = 0.09:

| Fr | Regime observed | Status |
|---|---|---|
| 1 × 10⁻⁴ | rolling | ✅ |
| 1 × 10⁻³ | rolling | ✅ |
| 1 × 10⁻² | cascading (S-curved free surface appears) | ✅ |
| 5 × 10⁻² | cascading | ✅ |
| 0.1 | cascading | ✅ |
| 0.5 | cataracting (particles detach from the wall) | ✅ |
| 1.0 | transitioning to centrifuging | ✅ |
| 1.5 | centrifuging | ✅ |

**PASS: at least 3 regime transitions occur within a factor of 2 in Fr of the values
above,** with the qualitative signature matching (S-curve for cascading, detachment for
cataracting, bed pinned to wall for centrifuging).

**Important caveat — do not treat these as universal constants.** [Mellmann2001]'s
central result is a Bed Behaviour Diagram in which the transitions depend on Froude
number, filling degree, **and** wall friction coefficient — not on Fr alone. Published
Fr values for the same nominal transition differ substantially between studies for
exactly this reason. The comparison above is valid only because fill level (50%) and
wall condition (rough, particle-lined) are matched to [Sunday2020]. Any run that
changes fill level or wall friction is outside this target and must be reported
separately.

### Secondary check — dynamic angle of repose vs Froude number

Stronger than regime classification, because it is a continuous quantitative curve
rather than a categorical judgement. [Sunday2020] Table 3, d = 1.0 mm, μ_r = 0.09,
g = 9.81 m/s², 60 mm drum, 50% fill:

| Fr | ω (rpm) | Dynamic repose angle θ (deg) | Flowing layer thickness (particle diameters) | Status |
|---|---|---|---|---|
| 1 × 10⁻⁴ | 1.7 | 32.1 ± 0.6 | 4.2 | ✅ |
| 1 × 10⁻³ | 5.4 | 35.0 ± 0.6 | 4.9 | ✅ |
| 1 × 10⁻² | 17 | 41.4 ± 1.1 | 6.5 | ✅ |
| 5 × 10⁻² | 39 | 51.5 ± 0.9 | 8.2 | ✅ |
| 0.1 | 55 | 57.2 ± 0.7 | 9.3 | ✅ |

**PASS: simulated θ within ± 4° at each Fr, and monotonically increasing with Fr.**

The monotonic-increase requirement matters as much as the absolute values — it is the
harder thing to get right by accident, and it is what the experimental data of Brucks
et al. (2007) collapses onto when plotted against Fr.

**Honesty note carried into the writeup:** [Sunday2020] reports that their own
simulated angles run roughly 5–7° above the Brucks et al. experimental values, and
attribute this to a mismatch in material properties and to their particle-lined drum
wall being a stronger boundary condition than the sandpaper-lined experimental drum.
So this target validates against *another DEM code*, not directly against experiment.
That is a weaker claim and must be stated as such. Target 1 is the direct
experimental comparison.

### Rolling-friction sensitivity check

[Sunday2020] at Fr = 1 × 10⁻³, d = 1.0 mm:

| μ_r | θ (deg) | Status |
|---|---|---|
| 0 | 26.1 ± 0.6 | ✅ |
| 0.09 | 35.0 ± 0.6 | ✅ |

A ~9° swing from rolling friction alone. If your solver shows a much smaller
sensitivity, rolling resistance is probably not being applied correctly — check this
before anything else.

---

## Target 4 — Coefficient of restitution (Phase 1 unit test)

Two-particle head-on collision, no gravity, no friction.

**PASS: measured e within 2% of configured e, for e ∈ {0.5, 0.7, 0.9}.**
**PASS: measured e independent of impact velocity across 0.1 – 2.0 m/s (spread < 2%).**

[Sunday2020] §4.1.3 ran exactly this test with 100 values of e from 0 to 1 and found
negligible difference between input and measured e for both Hooke and Hertz models. A
Hertz–Mindlin implementation that cannot reproduce its own input restitution has a
damping-coefficient bug, and there is no point proceeding.

### Additional two-body tests worth stealing from [Sunday2020] §4

These are cheap, they run on the CPU backend, and each isolates one force term:

| Test | What it checks | Pass criterion |
|---|---|---|
| Stacked drop | Normal force only | 5 spheres dropped in vertical alignment come to rest stacked; no lateral drift or rotation |
| Oblique impact | Tangential force + Coulomb transition | For μ_s = 0.3, e = 1, the sliding-regime threshold angle is 64.54°; above it, post-collision spin ω′ = 3 rad/s and tangential restitution follows the analytic expression |
| Block sliding | Coulomb friction magnitude | v₀ = 5 m/s, μ_s = 0.5 → block travels 2.5484 m. Agreement to < 1 × 10⁻³ m |
| Rolling decay | Rolling resistance torque | Sphere given 1 m/s slides, then rolls, then stops; resistance torque constant and non-zero while moving |

The oblique-impact threshold angle formula: φ_th = arctan[ (7/2) μ_s (1 + e) ].

---

## Target 5 — Stiffness scaling justification

Young's modulus is scaled below the physical value to make the timestep tractable.
Standard practice, and it MUST be validated rather than assumed.

**Precedent.** [Sunday2020] used E = 70 MPa against a true glass value of ~70 GPa — a
reduction of three orders of magnitude — citing Chen et al. (2017), who found that E
can be reduced by at least three orders of magnitude before variations in tumbler flow
begin to appear. [Just2013] similarly found the dynamic angle of repose insensitive to
shear modulus across the range they tested (0.77 – 24 MPa, angles 38.4–38.8°), and
Ketterhagen (2011) reported that a 100× increase in shear modulus had no impact on
simulation results.

**PASS: repose angle and hopper discharge rate vary by less than 5% across
E ∈ {7 MPa, 70 MPa, 700 MPa}.**

If they do not, the scaling range is too aggressive and must be narrowed. Report the
sweep either way — a negative result here is publishable material.

**Overlap check (run on every validation case):**
**PASS: mean δ_n / particle radius < 1%, and max δ_n / radius < 5%.**

Excessive overlap is the failure mode that soft contacts produce: particles
interpenetrate, the bed behaves like a fluid, and the repose angle collapses. The
overlap check catches this even when the repose angle happens to look plausible.

---

## Target 6 — Tablet material properties (Phase 5 onward, not a validation)

Not a pass/fail validation — there is no equivalent calibration dataset for tablets.
This section fixes the application-material inputs and their provenance so the coating
study rests on measured values rather than software defaults.

**Source:** [Just2013]. Material is a Bayer GITS (gastrointestinal therapeutic system):
a round biconvex two-layer tablet with a nifedipine API layer, an osmotic blend layer,
and a diffusion-membrane coat, active-coated with candesartan cilexetil in a side-vented
lab drum coater.

| Property | Uncoated core | Active-coated | Status |
|---|---|---|---|
| Young's modulus (uniaxial compression) | 31.9 ± 0.8 MPa | ~31.8 MPa (no significant difference) | ✅ |
| Coefficient of restitution | 0.79 ± 0.04 | 0.80 ± 0.03 | ✅ |
| Tablet–tablet friction (measured) | ~0.5 | ~0.5 | ✅ |
| Tablet–steel friction (measured) | 0.15 | higher than uncoated | ✅ |
| Tablet–steel friction (used in simulation) | **0.5** | 0.5 | ✅ |
| Tablet–tablet friction (used in simulation) | **0.5** | **0.14** | ✅ |
| Dynamic angle of repose, experimental | **39°** | **27°** | ✅ |
| Glued-sphere representation | 8 spheres | 8 spheres | ✅ |

**The most important line in this table is the friction discrepancy, and it belongs in
the writeup.** [Just2013] measured tablet–steel friction as 0.15, put it into the
simulation, and got a slipping bed — qualitatively the wrong flow regime. They found a
minimum of 0.45 was needed to produce a cascading bed matching the experiment, and used
0.5. Their own conclusion is that the friction measurement method did not adequately
describe tablet motion in a perforated drum, and the coefficients had to be revised
against the dynamic angle of repose.

Two things follow:

1. **Cite the measured value, use the calibrated value, and say so.** Presenting 0.5 as
   "measured" would be wrong.
2. **The tablet-side of this project is calibrated, not validated.** Only the glass-bead
   work in Targets 1–3 is a genuine validation against independent experiment. Blurring
   that line is exactly the failure this document exists to prevent.

**Sensitivity ordering from [Just2013], useful for debugging:** the dynamic angle of
repose was insensitive to shear modulus and restitution across the ranges tested, and
strongly sensitive to friction. With tablet–steel friction at 0.1 the angle sat around
26–27° regardless of tablet–tablet friction; at 0.5–0.9 it rose to 38–42°. If your
tablet bed has the wrong angle, look at wall friction first.

---

## Explicitly NOT validated

Stating these protects the credibility of what is validated.

- **Cohesion / liquid bridging.** Not modelled. Real coating is a wet process and
  capillary bridges change flow behaviour; [Sunday2020] shows dynamic repose rising with
  granular Bond number. Our spray model deposits mass without wetting mechanics.
- **Air phase and droplet transport.** Kinematic spray only; no CFD coupling.
- **Drying, thermal effects, solvent evaporation.**
- **Particle breakage.** Cumulative collision energy is logged as a damage *proxy*, with
  no calibration to actual attrition rates.
- **Coating uniformity itself.** No experimental CV dataset is available to us, so the
  coating results are a **comparative study across configurations**, not an absolute
  prediction of any real process. Rankings between configurations are the claim;
  absolute CV values are not.
- **Non-spherical contact mechanics** until Phase 4 exists. Before then, tablets are
  volume-equivalent spheres and any shape-dependent result is not meaningful.
- **Anything at industrial scale.** [Boehling2016] simulated ~1.03 million tablets
  (290 kg). Our largest runs will be well below that; scale-dependent conclusions are
  out of scope.

---

## Checklist before Phase 3 begins

- [ ] Every ⚠️ value confirmed against the primary PDF
- [ ] `configs/material/glass_beads.yaml` matches the Target 1 property table exactly
- [ ] `configs/material/tablet_placebo.yaml` matches the Target 6 table, with the
      measured-vs-calibrated friction distinction recorded in comments
- [ ] Measurement protocols implemented as code, not as manual post-processing
- [ ] This file committed and pushed with a timestamp preceding all solver commits