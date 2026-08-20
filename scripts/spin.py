"""Spin one particle freely and watch the quaternion norm.

    python scripts/spin.py
"""

import numpy as np

from warp_dem import describe_device, resolve_device
from warp_dem.solver import Solver
from warp_dem.state import ParticleState

DT = 1e-5
OMEGA = np.array([3.0, -4.0, 5.0])  # rad/s, world frame


def rotate(q, v):
    u = np.asarray(q[:3], dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + float(q[3]) * v)


def main() -> None:
    device = resolve_device("auto")
    print(describe_device(device))

    state = ParticleState.allocate(1, device, radius=0.002, density=2500.0)
    state.set_angular_velocities([OMEGA.tolist()])
    solver = Solver(state, dt=DT, gravity=(0.0, 0.0, 0.0))

    axis = OMEGA / np.linalg.norm(OMEGA)
    rate = float(np.linalg.norm(OMEGA))
    x_hat = np.array([1.0, 0.0, 0.0])

    print(f"\n{'steps':>9} {'t [s]':>8} {'|q| - 1':>12} {'body-x error':>14}")
    for _ in range(10):
        solver.run(10_000)
        q = state.orientations()[0]
        theta = rate * solver.time
        q_exact = np.concatenate([axis * np.sin(theta / 2.0), [np.cos(theta / 2.0)]])
        err = np.linalg.norm(rotate(q, x_hat) - rotate(q_exact, x_hat))
        print(f"{solver.step_count:9d} {solver.time:8.3f} "
              f"{np.linalg.norm(q) - 1.0:12.3e} {err:14.3e}")

    print("\n|q| - 1 must stay flat. The body-x error grows: that is float32 angle "
          "accumulation, the same effect measured in docs/precision.md.")


if __name__ == "__main__":
    main()