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