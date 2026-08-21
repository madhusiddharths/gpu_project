# Floating-point precision

**Status:** float32 throughout. Decision reviewed at Phase 7 (mixed precision).
**Measured:** Block 6, M3 Pro (arm64), Warp 1.16.0, CPU backend.

## The observation

Integrating `x += v*dt` in float32 with a constant velocity accumulates a
systematic, one-directional position error. Zero gravity, v = (1, -2, 0.5) m/s,
dt = 1e-5 s, 10,000 steps:

| component | simulated | exact | abs error | rel error |
|---|---|---|---|---|
| x | 0.09999150 | 0.1 | 8.50e-06 | 8.50e-05 |
| y | -0.19998300 | -0.2 | 1.70e-05 | 8.50e-05 |
| z | 0.04999575 | 0.05 | 4.25e-06 | 8.50e-05 |

Reproduced exactly in pure NumPy float32, so it is arithmetic and not a solver
or Warp defect. float64 reduces the same error to 6e-15.

## Mechanism

`x + inc` rounds to the nearest representable float, an error of up to
`0.5 * ulp(x) = 0.5 * EPS * |x|`. Two consequences:

1. **The error scales with the coordinate, not the distance travelled.** A
   particle sitting at x = 1.0 m accumulates 8x the drift per step of one at
   x = 0.125 m.
2. **The roundings are correlated, not random.** With a near-constant increment
   the same rounding direction repeats, so error grows roughly linearly in step
   count rather than as sqrt(N). Measured error per step rose from 1.1e-10 at
   step 1,250 to 4.4e-9 at step 40,000, tracking the growth of ulp(x).

The representation error of dt itself (float32 1e-5 is short by 2.5e-8 relative)
contributes ~2.5e-9 over 10k steps and is not the cause.

## Rigorous bound

`accumulation_bound(max_coord, steps) = 0.5 * EPS * max_coord * steps`

Loose by 1-2 orders in practice, but it is an upper bound, so tests written
against it cannot fail spuriously across platforms. Used in
`tests/test_integrator.py` and, from Block 17, in the overlap diagnostics.

## Why float32 is nonetheless kept

DEM is memory-bound. float64 doubles the bytes moved for every particle field on
every kernel, which is close to doubling runtime — and demonstrating GPU
throughput is the project's central claim. Single precision is standard in GPU
DEM implementations.

## Where it could bite, and the plan

Drum geometry, |x| <= 0.15 m, dt ~ 1e-5 s after stiffness scaling, 10 s
physical time = 1e6 steps.

- Fully correlated (worst case): 0.5 * 1.19e-7 * 0.15 * 1e6 = **8.9 mm**.
  Larger than a 2 mm particle radius. Unacceptable if it occurred.
- Decorrelated (sqrt(N)): **8.9 um**, against a target contact overlap of
  ~20 um at delta/r < 1%.

Correlation requires a near-constant increment. In a tumbling bed the velocity
reverses continuously, so the realistic case is near the decorrelated end — but
that is an argument, not a measurement.

**Actions:**

1. Block 17 overlap diagnostics logs measured position drift alongside overlap,
   so the correlated/decorrelated question is answered with data.
2. Phase 7 mixed-precision experiment flips `precision.vec3` to `wp.vec3d` for
   positions only and measures both the drift and the bandwidth cost. The plan
   already anticipates this: "positions in FP32 can break long-running contact
   stability. If it breaks, that's a finding worth writing up."
3. If drift proves material, the standard mitigation is to store positions
   relative to a cell or domain origin so |x| stays small, which shrinks ulp
   without widening the type.
---

## Catastrophic cancellation in contact overlap (Block 10)

A second, independent float32 effect, unrelated to the accumulation drift above.
Accumulation drift grows with step count; this one is present on step one.

### The observation

The static Hertz force test failed at δ/R = 1e-4 with 1.76e-3 relative error,
while the same code at δ/R = 1e-3 and 1e-2 passed at 2e-4. **The error scales as
1/δ.**

### Mechanism

```
δ = (r_i + r_j) − |x_j − x_i|
```

subtracts two lengths of order 4e-3 m to leave one of order 2e-7 m. Each operand
carries absolute error up to `ulp = EPS·|x|`; subtraction preserves that absolute
error while destroying almost all the significant digits. Relative precision of
the result is therefore inflated by `magnitude / difference`:

```
relative error in δ  ≈  EPS · 2R / δ
relative error in F  ≈  1.5 · EPS · 2R / δ        (F ~ δ^1.5)
```

Predicted 3.6e-3 at δ/R = 1e-4; observed 1.8e-3. The bound holds with ~2x margin.

Confirmed not to be a solver defect: a bug would not scale with contact depth,
and the deeper contacts were accurate to 1/10 and 1/100 of the error.

### Bound

`cancellation_bound(magnitude, difference, exponent) = exponent · EPS · magnitude / difference`

Companion to `accumulation_bound`, and used the same way — tests derive their
tolerance rather than declaring one. Used in `tests/test_hertz.py`.

### Two consequences

1. **Contact force precision degrades as contacts get shallower**, and stiffer
   materials have shallower contacts. Stiffness scaling therefore has a second,
   quieter cost alongside the timestep: it makes contact forces noisier. A
   barely-touching contact is the least accurately resolved one in the bed.
2. **The mitigation is the same as for accumulation drift** — store coordinates
   relative to a cell or domain origin so |x| stays small, shrinking ulp without
   widening the type. That one change addresses both effects, which strengthens
   the case for it in Phase 7.

---

## Correlation length measured — the §5.1 question, answered (Block 17)

The section above leaves the decisive question open: are successive roundings
correlated? `scripts/drift_probe.py` measures it.

### Method

Rather than a float64 reference run (see below for why not), measure the
**correlation length** L — the mean number of steps a velocity component keeps
its sign. If roundings stay correlated over runs of L steps, the drift is a
random walk over N/L runs of size L·δ:

```
drift ~ delta * L * sqrt(N/L) = delta * sqrt(L*N)
```

Enhancement over the fully decorrelated estimate is therefore **sqrt(L)**. The
model reduces correctly to both limits (L = 1 gives sqrt(N); L = N gives N); the
middle is an interpolation rather than a theorem.

### Result — the optimistic reading does NOT survive

128 particles poured into a box, 30,000 steps, M3 Pro CPU backend:

```
mean velocity sign-run : 721.5 steps  (2.4% of the run)
sqrt(L)                : 26.9x
```

Extrapolated to drum geometry (|x| <= 0.15 m, 1e6 steps, ~20 um overlap):

| case | drift | vs overlap |
|---|---|---|
| fully correlated (worst case) | 8940 um | 447x |
| **measured correlation length** | **240 um** | **12x** |
| fully decorrelated | 8.9 um | 0.45x |

**Drift lands 12x ABOVE the contact overlap.** Velocities do not reverse every
step; they reverse every ~721 steps, and sqrt(721) is most of the way from the
good case to the bad one.

**A drum is likely WORSE.** The measurement above is a settling transient. In a
drum a particle rides up with the wall at near-constant velocity before
avalanching, so its correlation length is set by the circulation period — of
order 1 s, or 150,000 steps at dt = 6.6 us. Treat 12x as a lower bound.

### Revised action

The mitigation moves **from Phase 7 to Phase 5**: store positions relative to a
cell or domain origin so |x| stays small. This shrinks ulp without widening the
type, so it costs no extra bandwidth — which matters because the performance
claim rests on staying memory-bound. It also fixes the Block 10 cancellation
finding, since both are consequences of large coordinates.

Superseded: the earlier note that "the realistic case is near the decorrelated
end". That was an argument; this is a measurement, and it disagrees.

---

## Correction: flipping precision is NOT a one-line change (Block 17)

`precision.py` gained a `WARP_DEM_PRECISION` environment override, and it does
switch every array dtype, kernel annotation and host-side NumPy dtype correctly.
It does **not** yield a working float64 solver.

Warp pins bare float literals in kernel bodies to float32 and then refuses to mix
them with float64 operands. `half = 0.5 * dt` fails as soon as `dt` is a double;
fixing `integrate.py` moved the failure to `forces.py`, and it cascades — every
literal in every kernel needs an explicit `scalar()` cast.

The architectural claim (one place decides precision) holds. The convenience
claim (one line) does not. Remaining work is mechanical, and belongs to the
Phase 7 mixed-precision experiment.
