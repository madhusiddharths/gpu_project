"""Persistent per-contact state — the tangential displacement history.

THE PROBLEM THIS SOLVES

Everything built up to Block 11 is stateless. Hand the solver two positions and
two velocities and it returns a force; nothing is remembered between steps.
Tangential friction is not like that. Static friction is a SPRING: two grains
resting on each other deform elastically at their contact, and the restoring
force depends on how far they have crept since they first touched — not on how
fast they are moving now. That accumulated slip has to survive from one timestep
to the next, be rotated as the pair reorients, and be destroyed the instant the
pair separates.

Friction that resets every step is not static friction. It is viscous drag with
a friction coefficient painted on, and the visible symptom is a pile that slumps
too flat. That is the single most commonly cited cause of a failed
angle-of-repose validation.

WHY NOT ATTACH HISTORY TO THE PAIR LIST

The obvious move is to store one displacement per entry in the Block 9 pair
list. It does not work: that list is rebuilt from scratch every step, and slots
are claimed with an atomic counter, so pair (i, j) lands in a different index
each time. Its ORDER is nondeterministic by construction — contacts.py says so
explicitly. There is no stable address to attach state to.

THE DESIGN: PER-PARTICLE SLOTS, KEYED ON THE LOWER INDEX

Each particle owns K fixed slots. Contact (i, j) with i < j is stored in one of
particle i's slots. Three properties follow, and they are the entire reason the
hardest block in the project is tractable:

1. SLOT i*K + c IS WRITTEN ONLY BY THREAD i. No compare-and-swap, no atomics on
   history at all. The only atomics left in the force kernel are the ones
   accumulating force onto the PARTNER, which is unavoidable.
2. CARRY-FORWARD IS A LOOKUP, NOT A SORT. Two buffers: scatter this step's
   partners into the new one, then for each new slot search the old slots of the
   SAME particle for the same partner and copy the displacement across. Only K
   entries are ever searched, and they are contiguous.
3. SEPARATION DESTROYS HISTORY FOR FREE. A pair absent from the new list is
   simply never copied. No eviction pass, no stale-generation stamps, no
   reference counting — the thing that has to happen is the thing that happens
   when you do nothing.

Cost is K*K = 256 comparisons per particle, of which only the occupied slots
enter the inner loop. That is wasteful and deliberate, in the same spirit as the
naive O(N^2) detector: correct by inspection now, optimised in Phase 2 when the
grid supplies a bounded, sorted neighbourhood.

WHY THE LOWER INDEX

Storing each contact once, on the lower-indexed participant, keeps exactly one
authoritative copy of the displacement. Storing it on both would create two
copies that must agree, and they would drift apart the moment a float operation
was applied in a different order on each side. It also halves the memory.

The cost is a mild asymmetry: a particle that happens to be the lowest index
among its neighbours holds all of its contacts, so K must cover the full
coordination number rather than half of it.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects.
"""

import numpy as np
import warp as wp

from warp_dem.contacts import ContactList
from warp_dem.precision import vec3

#: Slots per particle. Random close packing gives a coordination number around
#: 6, rising toward 12 under compression. A particle holds only those contacts
#: whose partner has a higher index — on average half its neighbours — but in
#: the worst case all of them, so this is sized against the full coordination
#: number rather than half of it.
DEFAULT_SLOTS_PER_PARTICLE = 16

#: Sentinel for an unoccupied slot. -1 rather than 0 because 0 is a valid
#: particle index, and a sentinel that collides with real data is a bug waiting
#: for the right test case.
EMPTY = -1


class HistoryOverflow(RuntimeError):
    """Raised when a particle has more contacts than it has slots."""


@wp.kernel
def scatter_pairs_to_slots(
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    slot_partner: wp.array(dtype=wp.int32),
    slot_count: wp.array(dtype=wp.int32),
    slots_per_particle: wp.int32,
):
    """Distribute the flat pair list into per-particle slots.

    One thread per pair. The atomic is on a per-particle counter rather than a
    single global one, so contention is proportional to a particle's
    coordination number — around 6 — instead of to the whole pair list. That is
    the same total number of atomic operations as a global counter but spread
    across n addresses, which is what makes it cheap.

    The counter is incremented even when the slot is out of range, so the final
    value reports how many slots were actually needed. Same convention as
    ContactList: report the shortfall, do not merely report that there was one.
    """
    p = wp.tid()
    if p >= pair_count[0]:
        return

    i = pair_i[p]
    j = pair_j[p]
    c = wp.atomic_add(slot_count, i, 1)
    if c < slots_per_particle:
        slot_partner[i * slots_per_particle + c] = j


@wp.kernel
def carry_history_forward(
    old_partner: wp.array(dtype=wp.int32),
    old_disp: wp.array(dtype=vec3),
    new_partner: wp.array(dtype=wp.int32),
    new_disp: wp.array(dtype=vec3),
    slots_per_particle: wp.int32,
):
    """Copy each surviving contact's displacement into its new slot.

    One thread per PARTICLE, not per contact, which is what makes this
    race-free: thread i is the only writer of particle i's slots.

    The inner loop deliberately has no early exit. `break` on a match would
    diverge the warp for a saving of a few dozen comparisons on an oracle
    implementation that is going to be replaced in Phase 2 anyway. A fixed trip
    count keeps every thread in a warp in lockstep.

    A new slot with no match in the old buffer is zeroed, which is the correct
    behaviour for a contact that has just formed: it has not slipped yet. A
    contact that has SEPARATED never appears in the new buffer at all, so its
    displacement is discarded by never being read — no eviction pass needed.
    """
    i = wp.tid()
    base = i * slots_per_particle

    for c in range(slots_per_particle):
        j = new_partner[base + c]
        carried = vec3(0.0, 0.0, 0.0)
        if j >= 0:
            for s in range(slots_per_particle):
                if old_partner[base + s] == j:
                    carried = old_disp[base + s]
        new_disp[base + c] = carried


class ContactHistory:
    """Per-particle tangential displacement, persistent across timesteps.

    Double-buffered. `partner`/`disp` are the live buffers the force kernel
    reads and writes; `_next_*` receive the rebuild and are then swapped in.
    Swapping references rather than copying arrays makes the rebuild cost
    independent of how much history survived.
    """

    def __init__(
        self,
        n: int,
        device,
        slots_per_particle: int = DEFAULT_SLOTS_PER_PARTICLE,
    ):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if slots_per_particle <= 0:
            raise ValueError(
                f"slots_per_particle must be positive, got {slots_per_particle}"
            )

        self.n = n
        self.device = device
        self.slots_per_particle = slots_per_particle
        self.last_max_slots = 0

        size = n * slots_per_particle
        self.partner = wp.full(size, EMPTY, dtype=wp.int32, device=device)
        self.disp = wp.zeros(size, dtype=vec3, device=device)
        self._next_partner = wp.full(size, EMPTY, dtype=wp.int32, device=device)
        self._next_disp = wp.zeros(size, dtype=vec3, device=device)
        self.count = wp.zeros(n, dtype=wp.int32, device=device)

    def rebuild(self, contacts: ContactList, check_overflow: bool = True) -> int:
        """Re-key the history onto this step's contact list.

        Must be called after `contacts.detect` and before the force kernel, on
        every step. Skipping it on a step would leave the force kernel reading
        the previous step's partner list, which is not merely stale — it would
        apply particle i's history to whichever particle now occupies that slot.

        Returns:
            The largest number of contacts any single particle claimed, or -1
            when the check is skipped.
        """
        if contacts.n != self.n:
            raise ValueError(
                f"history sized for n={self.n}, contact list has n={contacts.n}"
            )

        self._next_partner.fill_(EMPTY)
        self.count.zero_()

        wp.launch(
            scatter_pairs_to_slots,
            dim=contacts.capacity,
            inputs=[
                contacts.pair_i,
                contacts.pair_j,
                contacts.pair_count,
                self._next_partner,
                self.count,
                self.slots_per_particle,
            ],
            device=self.device,
        )
        wp.launch(
            carry_history_forward,
            dim=self.n,
            inputs=[
                self.partner,
                self.disp,
                self._next_partner,
                self._next_disp,
                self.slots_per_particle,
            ],
            device=self.device,
        )

        # Swap references. The buffers that were live become the scratch for
        # the next rebuild, so no allocation happens inside the simulation loop
        # — there is no device-side allocator to call even if we wanted one.
        self.partner, self._next_partner = self._next_partner, self.partner
        self.disp, self._next_disp = self._next_disp, self.disp

        if not check_overflow:
            return -1

        needed = int(self.count.numpy().max()) if self.n else 0
        if needed > self.slots_per_particle:
            raise HistoryOverflow(
                f"a particle claimed {needed} contacts but only "
                f"{self.slots_per_particle} slots exist per particle.\n"
                f"Contacts beyond the slot limit were DROPPED, and dropping a "
                f"contact silently deletes its tangential history — the pair "
                f"would keep colliding but lose its static friction.\n"
                f"Note this counts contacts stored on the LOWER-indexed "
                f"participant, so a particle that is the lowest index among all "
                f"its neighbours holds every one of its contacts.\n"
                f"Either the packing is denser than expected, or particles are "
                f"overlapping badly. Check the overlap diagnostics before simply "
                f"raising slots_per_particle."
            )
        self.last_max_slots = needed
        return needed

    def partners(self) -> np.ndarray:
        """Device-to-host read, shape (n, slots). Diagnostics and tests only."""
        return self.partner.numpy().reshape(self.n, self.slots_per_particle)

    def displacements(self) -> np.ndarray:
        """Device-to-host read, shape (n, slots, 3). Diagnostics and tests only."""
        return self.disp.numpy().reshape(self.n, self.slots_per_particle, 3)

    def as_map(self) -> dict[tuple[int, int], np.ndarray]:
        """History as {(i, j): displacement}, order-independent.

        The correct unit of comparison in a test. WHICH slot a contact occupies
        depends on the order the atomic counter handed out indices, so it is
        nondeterministic; which displacement belongs to which PAIR is not.
        """
        partners = self.partners()
        displacements = self.displacements()
        out = {}
        for i in range(self.n):
            for c in range(self.slots_per_particle):
                j = int(partners[i, c])
                if j != EMPTY:
                    out[(i, j)] = displacements[i, c].copy()
        return out
