"""Minimal velocity-Verlet solver.

Body force, contact forces, and boundary forces. The step structure has not
changed since Block 6: every new force term slots into `compute_forces` between
the two integrator launches, which is what that method exists to make possible.

Force terms are OPTIONAL AND COMPOSABLE, and that is load-bearing for testing
rather than a convenience. A restitution test wants normal force and no walls;
the energy audit wants a frictionless model; the incline test wants walls and
friction but no particle-particle contact at all. Each of those is a Solver
with a different pair of arguments, not a different solver.
"""

from __future__ import annotations

import warp as wp

from warp_dem.integrate import initialise_forces, kick, kick_drift
from warp_dem.precision import vec3
from warp_dem.state import ParticleState
from warp_dem.timestep import TimestepBudget, assert_timestep_valid


class Solver:
    """Advances a ParticleState in time.

    The device is taken from the state, never chosen here. Device resolution
    happens in exactly one place: warp_dem.device.resolve_device.
    """

    def __init__(
        self,
        state: ParticleState,
        dt: float,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        budget: TimestepBudget | None = None,
        contact_model=None,
        boundary=None,
    ):
        """
        Args:
            budget: stability bounds from timestep.compute_budget. Optional
                because the Rayleigh and Hertz criteria bound CONTACT
                resolution; a ballistic run has no contacts and no bound. From
                Block 9, when contacts exist, every physical run passes one.
            contact_model: anything with `.apply(state, dt)` that accumulates
                particle-particle forces. None means non-interacting particles.
            boundary: anything with `.apply(state, dt)` that accumulates wall
                forces. None means an unbounded domain.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if budget is not None:
            assert_timestep_valid(dt, budget)

        self.state = state
        self.budget = budget
        self.contact_model = contact_model
        self.boundary = boundary
        self.dt = float(dt)
        # vec3 from the precision policy, not wp.vec3f: the kernels are
        # generated against whatever that alias resolves to.
        self.gravity = vec3(*(float(g) for g in gravity))
        self.time = 0.0
        self.step_count = 0

        # Velocity-Verlet's first half-kick consumes f(x_0). If this line is
        # missing, step 1 integrates with a zero force array and every
        # trajectory is wrong by half a timestep, forever. Tested explicitly.
        #
        # dt = 0 so that the tangential friction springs are EVALUATED but not
        # ADVANCED. No time has passed yet, so crediting every contact with a
        # step's worth of slip here would bias the friction state of any run
        # started from a configuration that is already in contact.
        self.compute_forces(dt=0.0)

    def compute_forces(self, dt: float | None = None) -> None:
        """Evaluate all forces and torques at the current positions.

        initialise_forces ASSIGNS the body force and clears torque, so it must
        stay first; everything after it accumulates with wp.atomic_add. Getting
        that order wrong does not crash — it silently drops gravity, or silently
        drops every contact force computed so far.
        """
        s = self.state
        step_dt = self.dt if dt is None else dt
        wp.launch(
            initialise_forces,
            dim=s.n,
            inputs=[s.force, s.torque, s.mass, self.gravity],
            device=s.device,
        )
        if self.contact_model is not None:
            self.contact_model.apply(s, step_dt)
        if self.boundary is not None:
            self.boundary.apply(s, step_dt)

    def step(self) -> None:
        s = self.state
        wp.launch(
            kick_drift,
            dim=s.n,
            inputs=[
                s.pos, s.vel, s.force, s.inv_mass,
                s.orient, s.omega, s.torque, s.inv_inertia,
                self.dt,
            ],
            device=s.device,
        )
        self.compute_forces()
        wp.launch(
            kick,
            dim=s.n,
            inputs=[
                s.vel, s.force, s.inv_mass,
                s.omega, s.torque, s.inv_inertia,
                self.dt,
            ],
            device=s.device,
        )
        self.time += self.dt
        self.step_count += 1

    def run(self, steps: int) -> None:
        """Advance `steps` timesteps.

        Kernel launches are asynchronous on CUDA — `wp.launch` queues work and
        returns immediately. The synchronize is what makes wall-clock timing
        around this call meaningful, and it is why timing an unsynchronized
        launch is the classic way to measure a fictitious speedup.
        """
        for _ in range(steps):
            self.step()
        wp.synchronize_device(self.state.device)