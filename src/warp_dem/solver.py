"""Minimal velocity-Verlet solver.

Block 6 scope: body force only. Contact forces enter at Block 9; wall contacts
at Block 15. The step structure does not change when they do — new force
kernels slot into `compute_forces` between the two integrator launches.
"""

from __future__ import annotations

import warp as wp

from warp_dem.integrate import initialise_forces, kick, kick_drift
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
    ):
        """
        Args:
            budget: stability bounds from timestep.compute_budget. Optional
                because the Rayleigh and Hertz criteria bound CONTACT
                resolution; a ballistic run has no contacts and no bound. From
                Block 9, when contacts exist, every physical run passes one.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if budget is not None:
            assert_timestep_valid(dt, budget)

        self.state = state
        self.budget = budget
        self.dt = float(dt)
        self.gravity = wp.vec3f(*(float(g) for g in gravity))
        self.time = 0.0
        self.step_count = 0

        # Velocity-Verlet's first half-kick consumes f(x_0). If this line is
        # missing, step 1 integrates with a zero force array and every
        # trajectory is wrong by half a timestep, forever. Tested explicitly.
        self.compute_forces()

    def compute_forces(self) -> None:
        """Evaluate all forces and torques at the current positions.

        initialise_forces ASSIGNS, so it must stay first. Contact kernels added
        later accumulate and go after it.
        """
        s = self.state
        wp.launch(
            initialise_forces,
            dim=s.n,
            inputs=[s.force, s.torque, s.mass, self.gravity],
            device=s.device,
        )

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