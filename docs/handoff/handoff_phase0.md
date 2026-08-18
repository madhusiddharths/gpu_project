# warp-dem — project status and handoff

**Last updated:** 2026-08-16
**Purpose:** hand-off context for continuing this project in a new conversation.
Read this first, then `gpu_dem_project_plan.md` for the full original plan and
`docs/validation_targets.md` for the pre-registered validation targets.

---

## 1. Who I am and how to teach me

- **GPU/parallel programming:** none. CUDA, kernels, warps, atomics are all new.
- **Physics/numerical methods:** rusty. Treat me as a beginner on mechanics and integrators.
- **Time available:** 1–2 hours per day.
- **Goal:** learn properly, not ship fast. This is a portfolio + learning project.
- **Editor:** Sublime Text (not VS Code). Mac terminal + SSH for remote.

### Required block format

Work is delivered in **blocks of 60–90 minutes**, not days. Every block follows:

```
1. Concept   (~15 min) — the physics or GPU idea, from first principles
2. Build     (~50 min) — exact files, exact code, exact terminal commands
3. Verify    (~10 min) — a check that proves it works, then git commit
4. Interview (~10 min) — 2–3 questions an interviewer would ask about what
                          was just built, with answers
```

Every block ends with something committed and working. Nothing half-finished
carries over where avoidable.

---

## 2. Plan changes made so far

### 2.1 Schedule: 11 weeks → ~16 weeks

The original plan assumed near-full-time work. At 1–2 hrs/day that is ~110 hours
against a plan scoped for ~300.

| | Original | Revised |
|---|---|---|
| Unit of work | 1 day | 1 block (60–90 min) |
| Cadence | daily | ~5 blocks/week, 2 rest days |
| Duration | 11 weeks | ~16 weeks |
| Finish | Nov 2026 | ~mid-December 2026 |
| Phase 4 (non-spherical) | in scope | **deferred — decide at week 10** |
| Sweep size | 100–200 configs | 30–50, expandable |

Block budget by phase:

| Phase | Blocks | Weeks | Notes |
|---|---|---|---|
| 0 — Setup + literature | 5 | 1 | **DONE** |
| 1 — Minimal solver | 14 | 3 | The hard one. Contact mechanics from scratch |
| 2 — Neighbor search | 9 | 2 | First real GPU work |
| 3 — Validation | 9 | 2 | Low active time, lots of overnight runs |
| 5 — Drum + spray | 9 | 2 | |
| 6 — In-situ analytics | 9 | 2 | Plays to my data-engineering background |
| 7 — Profiling | 7 | 1.5 | Remote, needs contiguous blocks |
| 8 — Benchmark + sweep | 6 | 1.5 | Mostly waiting on jobs |
| 9 — Package + write | 7 | 1.5 | |

**Never cut:** Phase 3 validation, Phase 7 Nsight profiling.

### 2.2 Budget and compute: Lightning is NOT the GPU

Original plan assumed $30–60 and Lightning free-tier GPU hours. Reality: only
~$5 of Lightning credits available, and a Lightning T4 costs ~4.50 credits/hour.
That is about one hour total.

**Revised compute strategy:**

| Source | Hours | Cost | Use for |
|---|---|---|---|
| **M3 Pro (local)** | unlimited | $0 | Phases 1, 3, 4, 5, 9 and all development |
| **Lightning Studio** | unlimited on CPU | $0 | Persistent Linux box; verify code builds on x86_64 |
| **Kaggle** | ~30/week | $0 | Phase 2 scaling benchmark, Phase 8 sweep. **Phone verified ✅** |
| **Colab** | a few/day | $0 | Quick kernel checks |
| **vast.ai / RunPod** | as needed | ~$0.20–0.50/hr | **Phase 7 only** — the sole path to Nsight (needs SSH) |

Revised total: **$10–15**, almost all of it Phase 7 profiling.

Keep the Lightning Studio on **CPU**. Do not accept the "Move studio to AWS"
migration prompt — it charges a transfer fee before any compute runs.

### 2.3 Nsight can be analysed locally

NVIDIA restored macOS arm64 support for the Nsight host GUIs. Workflow:
**capture** profiles on the rented GPU with the CLI (`nsys profile`, `ncu`),
rsync the `.nsys-rep` / `.ncu-rep` files down, and **analyse on the M3 Pro with
the GPU instance destroyed**. Host and target versions must match, so pin them.
This materially reduces Phase 7 GPU spend.

### 2.4 Materials restructured: validation vs application

The original config had only glass beads. Now three materials, with an explicit
distinction that matters for the credibility of the whole project:

- `configs/material/glass_beads.yaml` — **validation material** (`role: validation`).
  Used only in Phase 3. Decades of published experiments with characterised
  properties.
- `configs/material/tablet_placebo.yaml` — **application material** (`role: application`).
  Used in Phases 5–8. Default in `config.yaml`.
- `configs/material/tablet_coated.yaml` — late-batch state; smoother film, lower friction.

Properties moved under a nested `contact:` block. Sphere-equivalent radius for
tablets is **volume-matched**, not diameter-matched (diameter-matching roughly
doubles particle mass and wrecks bed dynamics).

**The line that must not be blurred:** glass beads are *validated* against
independent experiment. Tablets are *calibrated* — published tablet friction
coefficients are themselves fitted to observed bed motion, not measured. Coating
results are therefore a **comparative study across configurations**, not an
absolute prediction.

### 2.5 Validation targets are now filled in and committed

`docs/validation_targets.md` is complete with real numbers and citations,
committed with a timestamp preceding any solver code. Key sources:

- **Sunday et al. (2020)**, MNRAS, arXiv:2009.10448 — open access, the primary
  reference. Repose angle 25.2 ± 0.8° experimental; full glass-bead property
  table; Froude regime transitions; dynamic repose vs Fr; four two-body unit tests.
- **Just et al. (2013)**, AAPS PharmSciTech, PMC3581635 — open access. Tablet
  properties: E = 31.9 MPa, CoR 0.79, dynamic repose 39° (uncoated) / 27° (coated).
- **Beverloo (1961)** — hopper correlation, C ≈ 0.58, k ≈ 1.4. ⚠️ constants still
  need confirming against the primary PDF.
- **Mellmann (2001)** — flow-regime classification. ⚠️ needs primary PDF.
- **Zhou et al. (2002)** — repose sensitivity. ⚠️ needs primary PDF.

Three caveats recorded in that file and to be carried into the writeup:

1. **Rolling friction μ_r = 0.09 is calibrated, not measured.** Adopting it and
   then "validating" against repose would be circular. The non-circular test:
   fix it as an input, then require Targets 2 and 3 to pass with no further
   adjustment.
2. **Froude transitions are not universal constants.** Mellmann's Bed Behaviour
   Diagram makes them depend on fill degree and wall friction too. The target is
   valid only because fill level (50%) and wall condition are matched.
3. **Target 3 validates against another DEM code, not experiment.** Sunday et al.
   report their own angles running 5–7° above the Brucks et al. experimental
   values. Target 1 is the direct experimental comparison.

### 2.6 Architectural decisions locked in

- **`src/` layout** — forces tests to exercise the installed package, not the
  source folder.
- **One device resolver** (`src/warp_dem/device.py`). Nothing else in the
  codebase may hardcode a device. CI greps for violations.
- **`auto` falls back to CPU; explicit `cuda:0` never does.** A silent fallback
  would produce a "GPU benchmark" from a CPU run.
- **Parity tests compare within tolerance, never bit-exact.** Float addition is
  not associative and atomic accumulation order is nondeterministic, so identical
  cross-device output is neither expected nor required.
- **Provenance on every benchmark** (`scripts/device_report.py`): GPU, driver,
  library versions, git commit, and whether the tree was dirty.
- **Two lock files** — `requirements.lock.macos.txt` and `.linux.txt`. `pip freeze`
  output is platform-specific; merging them produces a file that installs nowhere.
- **Stiffness scaling is validated, not assumed** — Target 5 in the validation doc.

---

## 3. Phase 0 — COMPLETE

| Criterion | Status |
|---|---|
| Warp runs on CPU backend, Mac (arm) and Linux (x86_64) | ✅ |
| Repo scaffold, device resolver, Hydra config | ✅ |
| Materials with provenance | ✅ |
| `docs/validation_targets.md` committed pre-solver | ✅ |
| Remote environment + git loop verified | ✅ |
| Kaggle GPU unlocked (phone verified) | ✅ |
| CI green | ✅ |
| Warp example on actual GPU hardware | ⏸ **deferred to Kaggle, before Phase 2** |

The deferral is deliberate. Phase 1 is entirely CPU-backend physics; there is
nothing to run on a GPU yet. It gets cleared in one free Kaggle session alongside
Phase 2's scaling benchmark.

### Repo state

```
gpu_project/
├── .github/workflows/ci.yml      # install → ruff → pytest → device-guard grep
├── configs/
│   ├── config.yaml               # device: auto | cpu | cuda:0
│   ├── material/{glass_beads,tablet_placebo,tablet_coated}.yaml
│   └── geometry/box.yaml
├── docs/validation_targets.md    # pre-registered, sourced
├── scripts/
│   ├── show_config.py
│   └── device_report.py          # provenance capture
├── src/warp_dem/
│   ├── __init__.py
│   └── device.py                 # THE only place a device is resolved
├── tests/
│   ├── test_config.py
│   ├── test_device.py
│   └── test_device_parity.py     # skips without CUDA
├── check_env.py
├── pyproject.toml
├── requirements.txt              # both platforms
├── requirements-gpu.txt          # CUDA only: cupy, cudf, nvtx
├── requirements.lock.{macos,linux}.txt
└── gpu_dem_project_plan.md
```

Current test count: 13 passed, 5 skipped (parity tests skip without CUDA).

---

## 4. Environment quirks — will bite if forgotten

**Mac**
- Repo at `~/Documents/python/gpu_project`, venv at `.venv/`, Python 3.12 (Homebrew)
- Conda `(base)` auto-activates; run `conda config --set auto_activate_base false`
  if not already done. Always confirm `which python` points at `.venv/bin/python`.
- Run `ruff check src tests scripts` before every commit — CI lints before testing,
  so a green local `pytest` is not enough.

**Lightning Studio**
- **Only `/teamspace/studios/this_studio` persists.** Files elsewhere vanish on stop.
  Repo lives at `/teamspace/studios/this_studio/gpu_project`.
- **Venvs are forbidden** — one conda env per Studio (`cloudspace`). Install
  directly with pip; no activation needed.
- **`~` and SSH's `$HOME` differ.** The shell's `~` is the teamspace path; SSH
  reads keys from `/home/zeus/.ssh/`. Generate keys with the explicit absolute
  path or SSH won't find them.
- SSH key backed up to `/teamspace/studios/this_studio/.keys/`; restore to
  `/home/zeus/.ssh/` after a Studio restart. `.keys/` is gitignored.
- The Studio's `pip freeze` captures the whole base image — the Linux lock file
  carries a header saying so.

---

## 5. Phase 1 plan — Minimal working solver (14 blocks, ~3 weeks)

**[LOCAL ONLY. Zero GPU hours.]** Build the simplest thing that is physically
correct. Resist optimising. The naive O(N²) contact detection written here becomes
the correctness oracle for every optimisation in the rest of the project.

### Block sequence

| Block | Content | Ends when |
|---|---|---|
| 6 | Particle state as structure-of-arrays; first `@wp.kernel`; velocity-Verlet integrator | Free fall matches analytic trajectory |
| 7 | Quaternion orientation and angular velocity integration | Free rotation is stable; quaternion stays normalised |
| 8 | Rayleigh timestep criterion, asserted in code | Bad config fails loudly with a clear message |
| 9 | Naive O(N²) contact detection | Correct pair list vs a brute-force NumPy reference |
| 10 | Hertz normal force with viscous damping | Restitution matches config within 2% (**Validation Target 4**) |
| 11 | Two-body test battery from Sunday et al. §4 | Stacked drop, oblique impact, block sliding all pass |
| 12 | **Mindlin tangential force part 1** — contact-pair history storage | History survives across timesteps; destroyed on separation |
| 13 | **Mindlin tangential force part 2** — Coulomb clipping, history rotation | Particle on incline slides above friction angle, holds below |
| 14 | Rolling resistance torque | Repose angle responds to μ_r with roughly the ~9° sensitivity Sunday et al. report |
| 15 | Axis-aligned box boundaries with wall contacts | Particle settles on a plane; no jitter, no sinking |
| 16 | Energy conservation audit | Frictionless undamped config conserves energy over 10k steps |
| 17 | Overlap diagnostics | Mean δ/radius < 1% logged every run (**Validation Target 5**) |
| 18 | Hydra wiring — full solver runnable from a YAML | `python scripts/run.py material=glass_beads` works end to end |
| 19 | Phase 1 consolidation; run full suite on Lightning CPU (x86_64) | All five plan tests + two-body battery green on both machines |

### The five required unit tests (from the project plan, all run in CI)

1. Single particle in free fall matches the analytic trajectory
2. Two-particle head-on collision: measured restitution within 2% of input
3. Particle dropped on a plane reaches static equilibrium and stays (no jitter, no sinking)
4. Particle on an inclined plane: slides above the friction angle, holds below
5. Energy conserved in a frictionless, non-damped configuration

**Phase 1 done-when:** all five pass and restitution error is under 2%.

### Where I will get stuck — flagged in advance

- **Blocks 12–13, Mindlin tangential friction.** The single hardest thing in the
  project. Everything else in the solver is stateless — hand it two positions and
  velocities, get a force. Tangential friction is not: every active contact pair
  carries a running displacement vector that must be updated each step, rotated as
  the pair reorients, and destroyed the instant the pair separates. Per-pair
  persistent state in a structure that is rebuilt every timestep. Budget two full
  blocks and over-explain.
- This is also the #1 cause of failed Phase 3 validation. Symptom: a pile that
  slumps too flat, because static friction that resets every step is not static
  friction.

### Concepts to teach during Phase 1

Structure-of-arrays and memory coalescing · `wp.tid()` and launch dimensions ·
velocity-Verlet and why not Euler (symplectic, bounded energy error) ·
quaternions and gimbal lock · Hertz contact and the δ^1.5 exponent · coefficient
of restitution via damping · Mindlin tangential history and Coulomb clipping ·
rolling resistance as a correction for unmodelled shape · Rayleigh timestep and
why dt ≈ 1 µs · stiffness scaling as a validated practitioner technique

---

## 6. Next action

Start **Block 6** — particle state as structure-of-arrays and the velocity-Verlet
integrator. Local, on the M3 Pro, CPU backend.