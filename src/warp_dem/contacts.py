"""Naive O(N^2) contact detection — the correctness oracle.

DELIBERATELY the slowest possible algorithm. Its value is that it has no
spatial data structure and therefore none of the quiet failure modes a grid
has: no boundary binning errors, no double-counted straddling pairs, no
contacts missed because a particle exceeded the cell size. It is correct by
inspection, so from Phase 2 onward every optimisation is diffed against it and
any discrepancy belongs to the optimisation.

An explicit pair LIST, rather than on-the-fly force computation, because
Blocks 12-13 attach persistent tangential-displacement history to each active
contact. Per-pair state requires an identified pair.

Each unordered pair appears exactly once (j > i), by Newton's third law.

ORDERING IS NONDETERMINISTIC. Slots are claimed with an atomic counter, so
which thread gets which slot depends on scheduling. The list contents are
deterministic; the order is not. Compare pair SETS, never pair arrays.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects.
"""

import numpy as np
import warp as wp

from warp_dem.precision import scalar, vec3
from warp_dem.state import ParticleState

#: Contacts budgeted per particle. Random close packing of spheres has a
#: coordination number around 6, rising toward 12 under compression; 16 is
#: headroom. Pairs are stored once, so capacity is n * this / 2.
DEFAULT_CONTACTS_PER_PARTICLE = 16


class ContactOverflow(RuntimeError):
    """Raised when more contacts were found than the pair list can hold."""


@wp.kernel
def detect_contacts_naive(
    pos: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    n: wp.int32,
    capacity: wp.int32,
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
):
    """One thread per particle; each tests every higher-indexed partner.

    Squared distances throughout — a square root would run on the GPU's
    special-function unit, and this comparison is the hottest line in the naive
    solver at N^2 executions per step.

    Contact is a STRICT inequality: exactly touching produces zero overlap and
    therefore zero Hertz force, so admitting it would consume a slot and a
    tangential-history entry for nothing.

    The counter is incremented even when the slot is out of range, so the final
    value reports how much capacity was actually needed rather than merely that
    it was exceeded.

    Load imbalance is accepted here: thread 0 iterates n-1 times and the last
    thread zero times. Within a warp the trip counts differ by at most 32, so
    this is grid-level imbalance rather than warp divergence. The oracle is not
    optimised; Phase 2 removes the question entirely.
    """
    i = wp.tid()
    xi = pos[i]
    ri = radius[i]

    for j in range(i + 1, n):
        d = pos[j] - xi
        cutoff = ri + radius[j]
        if wp.dot(d, d) < cutoff * cutoff:
            slot = wp.atomic_add(pair_count, 0, 1)
            if slot < capacity:
                pair_i[slot] = i
                pair_j[slot] = j


class ContactList:
    """Preallocated storage for detected contact pairs.

    There is no device-side allocator, so capacity is fixed before launch and
    overflow is detected afterwards.
    """

    def __init__(
        self,
        n: int,
        device,
        contacts_per_particle: int = DEFAULT_CONTACTS_PER_PARTICLE,
        capacity: int | None = None,
    ):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.n = n
        self.device = device
        self.capacity = capacity if capacity is not None else max(
            64, (n * contacts_per_particle) // 2
        )
        self.pair_i = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        self.pair_j = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        self.pair_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.last_count = 0

    def detect(self, state: ParticleState, check_overflow: bool = True) -> int:
        """Rebuild the pair list from current positions.

        Args:
            check_overflow: reads the counter back to the host, which forces a
                device synchronisation and stalls the asynchronous launch
                pipeline. Correct for development and for every test; in a
                production loop this should be checked every N steps rather
                than every step. Flagged for Phase 7.

        Returns:
            The number of contacts found, or -1 when the check is skipped.
        """
        if state.n != self.n:
            raise ValueError(f"list sized for n={self.n}, state has n={state.n}")

        self.pair_count.zero_()
        wp.launch(
            detect_contacts_naive,
            dim=self.n,
            inputs=[
                state.pos,
                state.radius,
                self.n,
                self.capacity,
                self.pair_i,
                self.pair_j,
                self.pair_count,
            ],
            device=self.device,
        )

        if not check_overflow:
            return -1

        found = int(self.pair_count.numpy()[0])
        if found > self.capacity:
            raise ContactOverflow(
                f"found {found} pairs, capacity is {self.capacity} "
                f"({2.0 * found / self.n:.1f} contacts per particle, budgeted "
                f"{2.0 * self.capacity / self.n:.0f}).\n"
                f"NOTE: a pair is two contacts, one for each participant. The "
                f"list stores PAIRS; the per-particle figures above are CONTACTS.\n"
                f"Either the packing is denser than expected, or particles are "
                f"overlapping badly because the timestep or contact stiffness is "
                f"wrong. Check the overlap diagnostics before simply raising "
                f"contacts_per_particle."
            )
        self.last_count = found
        return found

    def pairs(self) -> np.ndarray:
        """Device-to-host read of the active pairs, shape (m, 2).

        Order is nondeterministic. Diagnostics and tests only.
        """
        m = self.last_count
        return np.stack([self.pair_i.numpy()[:m], self.pair_j.numpy()[:m]], axis=1)

    def pair_set(self) -> set[tuple[int, int]]:
        """The pairs as an order-independent set — the correct unit of comparison."""
        return {(int(a), int(b)) for a, b in self.pairs()}