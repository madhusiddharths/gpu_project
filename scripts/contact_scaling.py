"""Measure how naive contact detection scales. The baseline Phase 2 beats.

    python scripts/contact_scaling.py
"""

import time

import numpy as np
import warp as wp

from warp_dem import describe_device, resolve_device
from warp_dem.contacts import ContactList
from warp_dem.state import ParticleState

RADIUS = 0.002
DENSITY = 2500.0
PACKING = 0.25  # solid fraction; fixed so contacts per particle stays comparable


def main() -> None:
    device = resolve_device("auto")
    print(describe_device(device))

    rng = np.random.default_rng(0)
    print(f"\n{'N':>8} {'pairs':>10} {'per particle':>13} {'ms/detect':>11} "
          f"{'ms / N^2':>12}")

    for n in (500, 1000, 2000, 4000, 8000):
        side = (n * (4.0 / 3.0) * np.pi * RADIUS**3 / PACKING) ** (1.0 / 3.0)
        positions = rng.uniform(0.0, side, size=(n, 3)).astype(np.float32)

        state = ParticleState.allocate(n, device, RADIUS, DENSITY)
        state.set_positions(positions)
        contacts = ContactList(n, device)

        contacts.detect(state)  # warm up: first launch pays compilation
        reps = 5
        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(reps):
            found = contacts.detect(state, check_overflow=False)
        wp.synchronize_device(device)
        elapsed = (time.perf_counter() - start) / reps

        found = contacts.detect(state)
        print(f"{n:8d} {found:10d} {found / n:13.2f} {elapsed * 1e3:11.3f} "
              f"{elapsed / n**2 * 1e9:12.4f}")

    print("\nThe last column is time normalised by N^2. If it is roughly flat, "
          "the cost is quadratic\nand Phase 2's grid has somewhere to go.")


if __name__ == "__main__":
    main()