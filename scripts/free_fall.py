"""Drop one particle, print simulated vs analytic height.

    python scripts/free_fall.py
"""

import numpy as np

from warp_dem import describe_device, resolve_device
from warp_dem.solver import Solver
from warp_dem.state import ParticleState

DT = 1e-5
G = -9.81
Z0 = 1.0


def main() -> None:
    device = resolve_device("auto")
    print(describe_device(device))

    state = ParticleState.allocate(1, device, radius=0.002, density=2500.0)
    state.set_positions([[0.0, 0.0, Z0]])
    solver = Solver(state, dt=DT, gravity=(0.0, 0.0, G))

    print(f"\n{'t [s]':>8}  {'z_sim [m]':>12}  {'z_exact [m]':>12}  {'error [m]':>12}")
    for _ in range(10):
        solver.run(1000)
        t = solver.time
        z_sim = float(state.positions()[0, 2])
        z_exact = Z0 + 0.5 * G * t**2
        print(f"{t:8.3f}  {z_sim:12.8f}  {z_exact:12.8f}  {abs(z_sim - z_exact):12.3e}")

    print(f"\nsteps: {solver.step_count}   max |error|: "
          f"{abs(float(state.positions()[0, 2]) - (Z0 + 0.5 * G * solver.time**2)):.3e} m")
    print(f"float32 epsilon at z~1 m: {np.finfo(np.float32).eps:.3e} m per operation")


if __name__ == "__main__":
    main()