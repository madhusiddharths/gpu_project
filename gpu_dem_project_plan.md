# GPU-Accelerated DEM for Granular Coating Processes

**Working title:** `warp-dem` — A GPU-native discrete element solver with in-situ analytics for rotating-drum coating processes

**Duration:** 11 weeks | **Hardware:** MacBook Pro M3 Pro (local dev) + rented NVIDIA GPU (scale, profiling, benchmarks)

---

## 1. Scope

### 1.1 The problem

Discrete Element Method (DEM) simulation is the standard tool for designing granular processes — pharmaceutical tablet coating, food seasoning drums, mixers, crushers, conveyors. In pharma it matters acutely: in *active coating*, the coating layer carries the API itself, so tablet-to-tablet coating variation is a dosage-uniformity problem, not a cosmetic one.

The industry's own literature states the constraint plainly: because of long simulation times, DEM coating investigations are usually restricted to small-scale equipment, which limits what can be concluded about full-scale production and leaves scale-up under-informed.

That constraint is the project. DEM contact resolution is embarrassingly parallel and maps unusually well to GPUs. The gap between "this should be fast on a GPU" and "practitioners still run small cases" is the space this project occupies.

### 1.2 What this project is

A DEM solver written from scratch in NVIDIA Warp, with a zero-copy in-situ analytics layer, validated against published experimental benchmarks, profiled with Nsight, and used to run a parameter sweep that produces an actual process recommendation.

### 1.3 In scope

- 3D DEM solver: soft-sphere contact, velocity-Verlet integration
- Hertz–Mindlin normal contact with tangential history and rolling resistance
- Spherical particles (core) and non-spherical particles (glued-sphere, superquadric as stretch)
- GPU uniform-grid neighbor search, hand-written
- Rotating drum geometry with parameterized baffles, triangle-mesh boundaries
- Kinematic spray/deposition model with shadowing
- In-situ on-device analytics: coating CV, mixing index, residence time, collision energy
- Physics validation against published experimental data
- CPU-vs-GPU benchmark on identical hardware
- Parameter sweep across drum speed × fill level × baffle geometry × spray rate

### 1.4 Explicitly out of scope

Stating these is part of the deliverable — it demonstrates scoping judgment, and it protects the timeline.

- **CFD–DEM coupling.** The air phase and droplet transport are modeled kinematically, not resolved. Full coupling is a separate multi-month project.
- **Drying, thermal effects, solvent evaporation.**
- **Particle breakage and attrition.** Collision energy is logged as a damage *proxy* only.
- **Cohesion / liquid bridging** beyond an optional simple capillary model.
- **Multi-GPU.** Single-device only; the scaling story is particle count, not device count.
- **A GUI.** Config files and scripts.

### 1.5 Deliverables

| # | Deliverable | Form |
|---|---|---|
| 1 | `warp-dem` solver | Public GitHub repo, pip-installable, CI, tests |
| 2 | Validation report | Notebook + section in README, with error bars |
| 3 | Performance report | Scaling curves, Nsight traces, roofline placement |
| 4 | In-situ analytics benchmark | Conventional vs in-situ: wall-clock and bytes written |
| 5 | Parameter sweep findings | Result table + the process recommendation |
| 6 | Renders | 2–3 short videos of drum flow and coating evolution |
| 7 | Technical writeup | Long-form blog post / arXiv preprint |

---

### 1.6 Hardware reality and the two-machine workflow

**The constraint.** Warp runs on Apple Silicon macOS, but CPU-only — GPU execution requires a CUDA-capable NVIDIA device. There is no CUDA emulation path for Apple Silicon worth pursuing: nothing translates CUDA to Metal for real compute, Nsight does not run, and emulated performance numbers would be meaningless for a project whose entire deliverable is *measured* performance.

**Why this is workable anyway.** Warp's device is a runtime parameter. The identical kernel source runs `device="cpu"` on the M3 Pro and `device="cuda:0"` on rented hardware, with no code change. Design for that from line one and the friction stays small.

**What the M3 Pro is genuinely good for** — this is more than you'd expect:

- All physics correctness work: contact model, integrator, tangential history, unit tests
- Small-N validation runs. Angle of repose at 20–50k particles is comfortable; hopper discharge at ~100k is an overnight job
- Algorithm development for the analytics layer in NumPy before porting to CuPy
- Rendering and visualization (PyVista and Blender both run well on Metal)
- Sweep result analysis, plotting, and the entire writeup

Unified memory is an asset here: 18–36 GB means CPU reference runs aren't memory-limited the way an 8 GB discrete card would be.

**What strictly requires NVIDIA hardware:**

- Any GPU timing number
- Nsight Systems / Nsight Compute profiling — the credibility layer, non-negotiable
- CuPy and cuDF (the in-situ analytics path)
- Scale runs above ~200k particles
- The parameter sweep

**Benchmark methodology — this now matters more, not less.** Never compare M3 Pro CPU against a rented GPU. Different architecture, different memory system, different compiler; a reviewer will discard the number and be right to. Run **both** the CPU and CUDA paths on the same rented instance and compare those. Same code, same machine, one flag different.

Your constraint forces a cleaner experimental design than most people manage. Say so explicitly in the writeup — it reads as methodological discipline rather than a limitation.

**Remote setup:**

| Use | Option | Cost |
|---|---|---|
| Persistent dev environment (files persist, VSCode attach) | Lightning AI Studios free tier | $0 |
| Quick kernel checks | Colab (Warp tutorial notebooks run there directly) | $0 |
| Longer validation runs | Kaggle, ~30 GPU-hrs/week | $0 |
| Profiling + final sweep (needs SSH + Nsight) | vast.ai / RunPod, 3090 or 4090 | ~$0.20–0.50/hr |

Realistic total: **$30–60** for the whole project, assuming 40–80 GPU-hours across profiling, benchmarks, and a full sweep. Persistence matters more than raw speed — sessions dying mid-run is what will actually waste your time, which is why a Studios-style environment beats bare notebooks.

**Sync loop, set up on day one.** Git push/pull as the source of truth, `rsync` for result artifacts. Do not hand-copy files; you will do this a hundred times.

---

## 2. Skills and tools

### 2.1 What you will actually learn

This is the point of the project. Grouped by what it lets you claim.

**GPU programming**
- Warp kernel authoring: `@wp.kernel`, `wp.struct`, launch dimensions, `wp.tid()`
- Structure-of-arrays memory layout and why it matters for coalescing
- Atomics and contention avoidance
- Occupancy, warp divergence, register pressure
- Device memory lifetime, avoiding host round-trips
- Zero-copy interop via `__cuda_array_interface__`

**Profiling and performance engineering**
- Nsight Systems: timeline analysis, kernel attribution, gap-finding
- Nsight Compute: achieved occupancy, memory throughput vs peak, roofline
- Spatial locality optimization (particle reordering — usually the single biggest win)
- Measuring honestly: same physics, same tolerances, same machine

**Numerical methods and physics**
- Contact mechanics: Hertz normal, Mindlin–Deresiewicz tangential, history-dependent friction
- Velocity-Verlet integration and symplectic stability
- Rayleigh timestep criterion and why your timestep is what it is
- Stiffness scaling as a validated practitioner technique
- Spatial hashing, counting sort, parallel prefix scan

**Data engineering (your existing strength, applied on-device)**
- Streaming/online statistics that never materialize a full time series
- cuDF summary tables, CuPy reductions
- Sweep orchestration and experiment tracking

### 2.2 Stack

| Layer | Tool | Runs on | Note |
|---|---|---|---|
| Kernels / solver | **NVIDIA Warp** | Both | `pip install warp-lang`; same source, device flag switches |
| Array ops | **CuPy** | Remote | Zero-copy from Warp arrays. Prototype the math in NumPy locally |
| Tabular analytics | **cuDF** | Remote | Per-run and sweep-level summaries |
| Profiling | **Nsight Systems**, **Nsight Compute** | Remote only | Free; the credibility layer. Needs SSH, so vast.ai/RunPod not Colab |
| Geometry | **trimesh**, `wp.Mesh` (BVH) | Both | Don't hand-roll BVH — not the novel part |
| Rendering | **PyVista**, optional Blender | **Local** | M3 Pro handles this well |
| CPU reference | **Warp CPU backend** | **Remote** for the benchmark | Must run on the same machine as the GPU path |
| External check | **YADE** or **LIGGGHTS** | Local | Optional cross-solver sanity check, indicative only |
| Config / sweeps | **Hydra** or OmegaConf | Both | Every run reproducible from a YAML |
| Tracking | **Weights & Biases** | Both | Essential here — sweep runs remotely, you analyze locally |
| Testing | **pytest**, GitHub Actions | Local + CI | Physics unit tests run on the CPU backend, so CI is free |

---

## 3. Execution plan

Ten phases. Each has a **done-when** criterion — do not advance without it. The two gates that determine whether this project succeeds are **Phase 2** (is it fast) and **Phase 3** (is it correct).

Each phase is tagged **[LOCAL]**, **[REMOTE]**, or **[SPLIT]**. Phases 0–5 are essentially all local, which is why this works: the physics thinking — the genuinely hard part — happens on the M3 Pro. GPU hours are spent on measurement, not development.

---

### Phase 0 — Grounding and setup  **[SPLIT]**
**Days 1–4**

- Install Warp locally, confirm it initializes on the CPU device, run the shipped examples on the CPU backend.
- **Stand up the remote environment on day one, not week seven.** Provision a Lightning Studio or a vast.ai instance, install Warp + CuPy + cuDF + Nsight, confirm `wp.get_cuda_devices()` sees the card, and run the same example there. Verify the git push/pull loop end to end.
- Write a `device` resolver in your config: one Hydra key selects `cpu` or `cuda:0`, and nothing else in the codebase ever hardcodes a device.
- Read and take notes on 4–5 sources: a DEM tablet-coating review, the Bayer/Adalat pan-coater study, one GPU-DEM implementation paper, and a validation-benchmark paper (angle of repose, Beverloo hopper flow).
- **Write down your validation targets now, with numbers, before writing any solver code.** Pre-registering the targets is what stops you from unconsciously tuning parameters until the answer looks right.
- Repo scaffold: package layout, `pyproject.toml`, pytest, GitHub Actions CI running tests on the CPU backend.
- Hydra config schema for material properties, geometry, and run settings.

**Done when:** a Warp example runs on your GPU, CI is green, and your validation targets are committed to the repo.

---

### Phase 1 — Minimal working solver  **[LOCAL]**
**Week 1–2**

Build the simplest thing that is physically correct. Resist optimizing. **Entirely local** — this phase is about physics, and the CPU backend is the right tool. Zero GPU hours.

1. Particle state as **structure-of-arrays**: position, velocity, orientation quaternion, angular velocity, force, torque, radius, mass, inertia. SoA from the start — retrofitting is painful.
2. Velocity-Verlet integrator kernel. Quaternion integration for rotation.
3. **Naive O(N²) contact detection.** Deliberately. This is your correctness oracle forever after — every optimization gets diffed against it.
4. Hertz normal force with viscous damping; Mindlin tangential force with accumulated tangential displacement history and Coulomb clipping; rolling resistance torque.
5. Axis-aligned box boundaries.
6. Timestep from the Rayleigh criterion; assert it in code so a bad config fails loudly.

**Unit tests (these run in CI):**
- Single particle in free fall matches the analytic trajectory
- Two-particle head-on collision: measured coefficient of restitution matches the input within 2%
- Particle dropped on a plane reaches static equilibrium and stays there (no jitter, no sinking)
- Particle on an inclined plane: slides above the friction angle, holds below it
- Energy is conserved in a frictionless, non-damped configuration

**Done when:** all five tests pass and restitution error is under 2%.

---

### Phase 2 — Neighbor search (performance gate)  **[SPLIT]**
**Week 3**

This is where the speed lives. Write and correctness-test the grid locally; **the scaling benchmark must run remotely.**

1. Uniform grid with cell size ≈ 2 × max particle radius.
2. Cell-index kernel → parallel prefix scan (`wp.utils.array_scan`) → counting sort into cell-sorted particle order.
3. Contact kernel iterating the 27-cell neighborhood.
4. **Write this yourself first**, then compare against Warp's built-in `wp.HashGrid` for both correctness and performance. Doing it in that order gets you the engineering credential *and* a reference implementation to check against — if yours is 3× slower than the built-in, you have something specific to go fix, and that investigation is itself good material.
5. Handle polydisperse radii (multi-level grid, or size the cell to the largest particle and accept the cost — document the tradeoff).

**Benchmark now [REMOTE]:** naive vs grid at 10k / 100k / 1M particles. Verify identical forces to float tolerance — this check runs locally at 10k first, then remotely at scale. Budget roughly 2–3 GPU-hours.

**Done when:** grid results match naive within tolerance, and scaling is near-linear in N rather than quadratic.

---

### Phase 3 — Physics validation (credibility gate)  **[LOCAL]**
**Week 4**

**Nothing downstream means anything if this fails.** Most portfolio simulation projects skip this step, which is exactly why they don't survive interview scrutiny.

Convenient fact: all three validations use particle counts that fit comfortably on the M3 Pro. Repose runs at 20–50k, hopper at 50–100k as an overnight job, drum regimes at 30–50k. **You can complete the single most important phase of this project without spending a cent on GPU time.**

Three independent validations:

1. **Angle of repose.** Pour particles from a funnel onto a flat plate, let them settle, fit the pile surface. Compare to published experimental values for a material with known properties. Target: within 2–3°.
2. **Hopper discharge.** Measure steady mass flow rate from an orifice; compare against the Beverloo correlation. Target: within 10%.
3. **Rotating drum flow regime.** Sweep Froude number and confirm the transitions — slumping → rolling → cascading → cataracting — occur at the documented ranges.

If validation fails, the usual culprits in order: tangential history not being carried correctly between timesteps, timestep violating Rayleigh, rolling resistance missing or mis-scaled, wall friction unset.

**Done when:** all three targets are met and the results are committed as a notebook with plots.

---

### Phase 4 — Non-spherical particles  **[LOCAL]**
**Week 5**

Tablets are biconvex; flakes and grains are irregular. Sphericity strongly changes flow behavior, so this matters physically, not just visually.

- **Primary path: glued-sphere (multi-sphere).** Represent each tablet as 4–8 rigid-bonded sub-spheres. Reuses your entire contact pipeline; you add rigid-body assembly, composite inertia tensors, and torque accumulation. Known cost: shape fidelity requires many sub-spheres, which multiplies the contact workload — measure and report that.
- **Stretch path: superquadrics.** One particle, one shape function, contact point via Newton iteration. Far fewer contact pairs, much harder kernel, and divergence-prone. Attempt only if Phase 4 finishes early.

**Validation:** repose angle for the non-spherical case must shift in the physically correct direction (higher than spheres) and by a plausible magnitude.

**Done when:** glued-sphere tablets tumble stably in a drum with correct qualitative flow and a validated repose angle.

---

### Phase 5 — Drum geometry and spray model  **[LOCAL]**
**Week 6**

1. Drum, baffles, and inlet as triangle mesh via trimesh; collision through `wp.Mesh` BVH. Use the built-in — geometry queries are not the interesting part of this project.
2. Rotation: rotate mesh transform each step (simpler and less error-prone than a rotating reference frame).
3. Parameterize baffle geometry — count, height, pitch angle — so it becomes a sweep dimension.
4. **Spray model:** cone emitted from a nozzle position. Ray-cast to determine which particles are exposed, handle shadowing by nearer particles, deposit coating mass proportional to projected area × dwell in the spray zone. Per-particle accumulated coating mass is the primary state variable you care about.
5. Assert global coating mass balance every step.

**Done when:** coating accumulates over time, mass balance holds, and rendered output shows a physically sensible cascading bed under the spray zone.

---

### Phase 6 — In-situ analytics  **[SPLIT]**
**Week 7–8**

**This is your differentiator and the part that connects to your existing data-engineering strength.** The conventional workflow writes VTK or HDF5 to disk, then reloads it into ParaView or pandas — a round-trip that often costs more wall-clock than the simulation and is the real reason practitioners run three configurations instead of two hundred.

Your particle state is already in device memory. Never move it.

**Split the work.** Develop and unit-test every metric in NumPy locally first — Welford, the Lacey index, the histogram accumulators. The math is the hard part and it's device-agnostic. Verify each against a slow reference implementation on small arrays. Only then port to CuPy/cuDF remotely, where the work reduces to plumbing plus verifying the zero-copy path. This is why the phase gets an extra week but almost no extra GPU hours.

1. Hand Warp arrays to CuPy zero-copy via `__cuda_array_interface__`. Verify no device-to-host copy occurs (Nsight will show you).
2. On-device metrics, computed every N steps:
   - **Coating mass CV** across particles — the primary quality metric
   - **Lacey mixing index** — how well the bed is mixing
   - **Residence time distribution** in the spray zone
   - **Cumulative collision energy** per particle — attrition/damage proxy
3. Use streaming/online formulations (Welford for variance, histogram accumulators) so you never hold the full time series.
4. cuDF for the per-run summary row; the sweep produces one tidy table.

**Benchmark explicitly and record both numbers:** conventional (write to disk → reload → analyze in pandas/PyVista) versus in-situ. Report wall-clock **and** bytes written.

**Done when:** you have the comparison table, and a Nsight trace showing zero host transfers in the analytics path.

---

### Phase 7 — Profiling and optimization  **[REMOTE ONLY]**
**Week 9**

The phase that turns this from "a simulation" into "performance engineering." **Nothing here can be done on the Mac.** Book a dedicated block on an SSH-accessible instance (vast.ai or RunPod — Colab and Kaggle won't give you Nsight). Budget 15–25 GPU-hours and treat it as focused work, not background.

1. **Nsight Systems first.** Get the timeline. Find which kernel dominates, and find the gaps between kernels — gaps usually mean synchronization or host round-trips you didn't know about.
2. **Nsight Compute on the top kernel.** Record achieved occupancy, memory throughput as a percentage of peak, and where it sits on the roofline. Establish whether you are memory-bound or compute-bound before optimizing — DEM is almost always memory-bound, and knowing that redirects your effort.
3. Optimizations in expected-payoff order:
   - **Particle reordering by cell** for spatial locality — usually the single biggest win
   - Kernel fusion to cut launch overhead and re-reads
   - Reducing atomic contention in force accumulation
   - Revisiting SoA field grouping so hot fields are read together
   - Mixed precision — attempt carefully; positions in FP32 can break long-running contact stability. If it breaks, that's a finding worth writing up.
4. **Keep a log of every optimization attempted, including the ones that made it slower or broke the physics.**

**Done when:** you can state your memory bandwidth utilization as a percentage of your GPU's peak and explain the gap.

The failed optimizations belong in the writeup. An honest "attempting X gave 1.4× but violated energy conservation at long timescales, so I reverted it" is worth more in an interview than a clean number, because it demonstrates you were measuring the right things.

---

### Phase 8 — Benchmark and sweep  **[REMOTE]**
**Week 10**

**Benchmarks:**
- Throughput (particle-steps/sec) vs particle count, from 10k to the largest that fits in VRAM
- **CPU vs GPU on the identical rented machine** — same code, Warp device flag flipped, both runs on the cloud instance. Your M3 Pro numbers never appear in this comparison. State the methodology explicitly in the writeup; it's a point in your favor.
- Optional, and interesting: report M3 Pro CPU throughput as a *separate* line item labeled as a different architecture. Useful context for readers, never mixed into the speedup claim.
- Optional external reference: same case in YADE or LIGGGHTS, noting that solver differences make it indicative rather than exact
- Time-to-solution for one realistic production-scale run

**Sweep:**
- Dimensions: drum speed × fill level × baffle configuration × spray rate
- Latin hypercube if the space is large, full factorial if small
- Target 100–200 configurations; 20 is still a legitimate study if time is short
- Every run reproducible from its Hydra config; results tracked in W&B — non-negotiable here, since runs execute remotely and you analyze locally
- Output: a Pareto front of coating uniformity (CV) against throughput and cumulative collision energy
- Budget 20–40 GPU-hours. Launch the sweep as a detached batch job with checkpointing so an instance dying doesn't cost you the whole run.

**Done when:** you can name a specific configuration and quantify why it wins.

---

### Phase 9 — Package and communicate  **[LOCAL]**
**Week 11**

Fully local. Rendering, plotting, and writing all run well on the M3 Pro.

- **README with the headline numbers in the first screen.** Nobody scrolls.
- 2–3 rendered videos: bed flow, coating evolution, a baffle comparison side-by-side.
- Technical writeup: motivation → method → validation → performance → findings → what didn't work.
- Clean install path, one-command reproduction of the validation suite.
- Distribution: GPU MODE Discord, r/CFD, LinkedIn, and the Warp GitHub discussions. Consider an arXiv preprint — it pairs well with your existing IEEE paper.
- Draft the resume bullets while the numbers are fresh.

---

## 4. Results to report

Define these now so you build toward them. Fill in the blanks as you go.

### 4.1 Performance
- Throughput: ___ million particle-steps/sec at ___ particles
- GPU vs CPU speedup on identical hardware: ___×
- Scaling behavior from 10k → ___ particles
- Achieved memory bandwidth: ___% of device peak
- Largest tractable simulation: ___ particles in ___ GB VRAM

### 4.2 In-situ analytics
- Conventional post-processing: ___ min/run, ___ GB written
- In-situ: ___ sec/run, ___ GB written
- Consequence: ___ configurations became feasible instead of ___

### 4.3 Physics validation
- Angle of repose: predicted ___° vs experimental ___°
- Hopper discharge vs Beverloo: ___% error
- Coefficient of restitution: ___% error
- Flow-regime transitions match documented Froude ranges: yes/no

### 4.4 Process finding
The one that makes it a real project rather than a benchmark:

> *"Baffle configuration B at 12 rpm reduced coating CV from 8.4% to 4.1% versus the baseline, at 6% lower throughput and 15% lower cumulative collision energy."*

### 4.5 The feasibility claim
> *"The 180-configuration sweep took N GPU-hours. The same study on CPU would have required M — which is precisely why the published literature restricts these investigations to small-scale equipment."*

---

## 5. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Superquadric contact detection consumes weeks | High | Glued-sphere is the primary path; superquadric is explicitly a stretch goal |
| Validation fails and you can't find why | Medium | Naive O(N²) oracle from Phase 1; check tangential history, Rayleigh timestep, wall friction in that order |
| Timestep so small runs take days | High (it's physics, not a bug) | Stiffness scaling within a validated range — standard practice; document the validation |
| Scope creep into CFD coupling | Medium | Out of scope, in writing, in section 1.4 |
| Renders eat a week | Medium | PyVista is enough. Blender/Omniverse only if Phase 9 has slack |
| **Local/remote sync friction burns hours** | **High** | Git as source of truth from day one; device is a config key, never hardcoded; rsync for artifacts only |
| **Code works on CPU, breaks on CUDA** | **Medium** | Race conditions and atomics behave differently under real parallelism. Run the full test suite on both devices at every phase boundary, not just at the end |
| **Cloud instance dies mid-sweep** | **Medium** | Checkpoint every N configs; W&B logging so partial results survive; prefer persistent-storage providers over ephemeral notebooks |
| Nsight learning curve on unfamiliar remote setup | Medium | Do one throwaway profiling session in Phase 2 on a trivial kernel, so Phase 7 isn't your first exposure |
| Cost creep from idle instances | Low | Always use hourly billing; destroy instances after each session; local dev means most days cost $0 |

---

## 6. Draft resume bullets

Write the real versions once you have numbers. These show the shape:

- Built a GPU-native discrete element solver in NVIDIA Warp (CUDA) simulating **N million** interacting particles for pharmaceutical tablet-coating processes, achieving **X×** speedup over the CPU implementation on identical hardware at **Y%** of theoretical memory bandwidth.
- Implemented hand-written GPU uniform-grid spatial hashing with parallel prefix-scan counting sort, reducing contact detection from O(N²) to near-linear and enabling **Z×** larger simulations than the naive baseline.
- Validated solver physics against published experimental benchmarks — angle of repose within **2°**, hopper discharge within **8%** of the Beverloo correlation — establishing quantitative credibility for downstream process studies.
- Designed a zero-copy in-situ analytics layer (Warp → CuPy → cuDF) computing coating uniformity, mixing index, and residence-time distributions on-device, cutting post-processing from **40 min to 90 s** per run and eliminating **200 GB** of intermediate disk I/O — making a **180-configuration** design sweep tractable.
- Profiled with Nsight Systems/Compute and applied spatial-locality reordering and kernel fusion for a further **X×** improvement; identified the process configuration reducing coating variability from **8.4% to 4.1%**.

---

## 7. Weekly checklist

| Week | Phase | Machine | Gate |
|---|---|---|---|
| 0 | Setup + literature | Split | Validation targets committed; remote env verified |
| 1–2 | Minimal solver | Local | 5 physics unit tests pass |
| 3 | Neighbor search | Split | Near-linear scaling, matches naive |
| 4 | **Validation** | Local | Repose within 3°, Beverloo within 10% |
| 5 | Non-spherical | Local | Tablets tumble stably, repose shifts correctly |
| 6 | Drum + spray | Local | Mass balance holds |
| 7–8 | In-situ analytics | Split | Zero host transfers in Nsight trace |
| 9 | Profiling | **Remote** | Can state bandwidth % and explain the gap |
| 10 | Benchmark + sweep | **Remote** | Named winning configuration |
| 11 | Package | Local | README, videos, writeup live |

**GPU spend by week:** $0 through week 2, a few hours in week 3, $0 in weeks 4–6, light in 7–8, then concentrated in weeks 9–10. Roughly 40–80 GPU-hours total, **$30–60**.

---

## 8. If you fall behind

Cut in this order, and do it deliberately rather than by drifting:

1. **Superquadrics** — glued-sphere is sufficient
2. **Sweep size** — 20 configurations instead of 180
3. **External CPU reference** (YADE/LIGGGHTS) — the Warp CPU backend comparison is the defensible one anyway
4. **Blender rendering** — PyVista output is fine
5. **Non-spherical entirely** — spheres validate cleanly and the performance story survives intact

**Never cut:** Phase 3 validation, or the Nsight profiling in Phase 7. Those two are what separate this from a physics demo, and they're the specific things a performance-engineering interviewer will ask about.
