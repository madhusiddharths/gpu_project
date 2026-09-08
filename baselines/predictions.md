# Cross-platform predictions — Block 19

Recorded 2026-09-08 on macOS arm64 (Apple M3 Pro), BEFORE any x86_64 run.
Reference numbers: `baselines/mac/`. Warp 1.16.0, numpy 2.5.2, Python 3.12.4.

## Why anything could differ

IEEE 754 requires correct rounding for + - * / and sqrt, so given the same
order of operations those are bit-identical across ARM64 and x86-64. Not
required, and therefore platform-dependent:

- libm functions (log, exp, pow) — no correct-rounding requirement, so Apple
  libm and glibc use different polynomials and disagree at ~0.5-1 ulp
- FMA contraction — one rounding instead of two, compiler and target dependent
- reduction order under atomics — float addition is not associative
- denormal flushing under fast-math

## Where those enter this code

| Site | Call | Bit-stable |
|---|---|---|
| `forces.normal_stiffness` (L85) | `wp.sqrt` | yes |
| `forces.hertz_normal_force` damping (L109) | `wp.sqrt` | yes |
| `forces.tangential_stiffness` (L299) | `wp.sqrt` | yes |
| `forces.hertz_elastic_energy` (L122) | **`wp.pow(delta, 2.5)`** | **no** |
| `materials.damping_ratio` (host) | `log` | no, but once per material |

The Hertz force law never calls pow: F_n = (2/3) S_n delta with
S_n = 2 E* sqrt(R* delta), so delta^1.5 is built from one sqrt and two
multiplies. The elastic *energy* does call pow. The energy audit is therefore
the most likely place for a platform difference to appear, and the force path
the least.

## Predictions

| Quantity | Mac | Prediction on x86_64 |
|---|---|---|
| pytest | 189 passed, 5 skipped | identical |
| Restitution e=0.90, spc=25 | 0.080% | 0.07-0.09% |
| Restitution e=0.50, spc=25 | 1.323% | within 0.05 pp |
| Velocity-independence spread | 0.002-0.008% | under 0.02% |
| requested spc=10 -> actual | 15 | 15 (integer, Rayleigh binds) |
| Ballistic drift, 40k steps | 6.3932e-04 m | within 20% |
| mean velocity sign-run | 721.5 steps | 500-1000, NOT 3 matching digits |
| sqrt(L) enhancement | 26.9x | 22-32x |
| drum drift vs overlap | 12.008x | 10-15x, verdict unchanged |
| overlap mean delta/R | 0.0449% | within 10%, Target 5 PASS |
| energy conservation test | pass | pass, tightest margin of any test |

## Falsifiers

- If the sign-run length reproduces to 3+ digits, the settling bed is more
  deterministic than the chaotic-amplification model assumes. Investigate
  before trusting the drift extrapolation.
- If the drum drift ratio lands outside 8-20x, then `drift ~ delta*sqrt(L*N)`
  (docs/precision.md 5.4) is weaker than claimed, and the Phase 5
  local-origin decision must be re-derived rather than inherited.
- If any test built only from + - * / sqrt fails, that is a bug, not a
  platform difference. Diagnose before widening any tolerance.
