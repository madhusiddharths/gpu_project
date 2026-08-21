"""Persistent contact history — Block 12.

The history machinery is tested here WITHOUT any tangential force, by writing
displacements by hand and checking they survive, transform and die correctly.
Separating the bookkeeping from the physics is deliberate: when the incline
test misbehaves in Block 13, this file already says whether history is the
suspect.

WHAT MUST BE TRUE
  1. A contact that persists keeps its displacement across a rebuild.
  2. A contact that separates loses it — and a later re-contact starts at zero.
  3. A new contact starts at zero rather than inheriting a slot's leftovers.
  4. None of the above depends on WHICH slot a contact happens to occupy.

Point 4 is the one that bites. Slot assignment goes through an atomic counter,
so it is nondeterministic; every assertion here compares pair->displacement
MAPS, never raw slot arrays.
"""

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device
from warp_dem.contacts import ContactList
from warp_dem.history import EMPTY, ContactHistory, HistoryOverflow
from warp_dem.state import ParticleState

RADIUS = 0.002
DENSITY = 2500.0


def _available_devices():
    devices = [resolve_device("cpu")]
    if wp.get_cuda_devices():
        devices.append(resolve_device("cuda:0"))
    return devices


@pytest.fixture(params=_available_devices(), ids=str)
def device(request):
    return request.param


def _chain(device, n, spacing):
    """n particles in a row along x, `spacing` apart."""
    state = ParticleState.allocate(n, device, RADIUS, DENSITY)
    state.set_positions([[k * spacing, 0.0, 0.0] for k in range(n)])
    return state


def _rebuild(state, contacts, history):
    contacts.detect(state)
    return history.rebuild(contacts)


def _write_history(history, values):
    """Stamp a displacement onto each (i, j) pair. Test scaffolding only."""
    partners = history.partners()
    disp = history.displacements()
    for i in range(history.n):
        for c in range(history.slots_per_particle):
            j = int(partners[i, c])
            if (i, j) in values:
                disp[i, c] = values[(i, j)]
    history.disp.assign(disp.reshape(-1, 3).astype(np.float32))


# ── slot bookkeeping ──────────────────────────────────────────────────────────


def test_slots_start_empty(device):
    history = ContactHistory(4, device)
    assert np.all(history.partners() == EMPTY)
    np.testing.assert_allclose(history.displacements(), 0.0)


def test_contacts_are_stored_on_the_lower_indexed_particle(device):
    """One authoritative copy per pair. Two copies would drift apart."""
    state = _chain(device, 4, spacing=1.9 * RADIUS)
    contacts = ContactList(4, device)
    history = ContactHistory(4, device)
    _rebuild(state, contacts, history)

    stored = history.as_map()
    assert set(stored) == {(0, 1), (1, 2), (2, 3)}
    for i, j in stored:
        assert i < j
    # Particle 3 is the highest index, so it owns nothing.
    assert np.all(history.partners()[3] == EMPTY)


def test_rebuild_reports_the_busiest_particle(device):
    state = _chain(device, 4, spacing=1.9 * RADIUS)
    contacts = ContactList(4, device)
    history = ContactHistory(4, device)
    assert _rebuild(state, contacts, history) == 1


def test_rebuild_rejects_a_mismatched_contact_list(device):
    history = ContactHistory(4, device)
    with pytest.raises(ValueError, match="sized for"):
        history.rebuild(ContactList(5, device))


def test_overflow_names_the_shortfall(device):
    """The message must say how many slots were needed, not merely 'too many'."""
    n = 8
    state = ParticleState.allocate(n, device, RADIUS, DENSITY)
    # All eight overlapping: particle 0 alone holds seven contacts.
    state.set_positions([[0.1 * RADIUS * k, 0.0, 0.0] for k in range(n)])
    contacts = ContactList(n, device)
    history = ContactHistory(n, device, slots_per_particle=3)

    contacts.detect(state)
    with pytest.raises(HistoryOverflow, match="claimed 7 contacts"):
        history.rebuild(contacts)


def test_overflow_can_be_skipped_to_avoid_a_sync(device):
    state = _chain(device, 4, spacing=1.9 * RADIUS)
    contacts = ContactList(4, device)
    history = ContactHistory(4, device)
    contacts.detect(state)
    assert history.rebuild(contacts, check_overflow=False) == -1


# ── persistence, the point of the module ──────────────────────────────────────


def test_history_survives_a_rebuild_while_the_contact_persists(device):
    state = _chain(device, 3, spacing=1.9 * RADIUS)
    contacts = ContactList(3, device)
    history = ContactHistory(3, device)
    _rebuild(state, contacts, history)

    stamped = {(0, 1): np.array([1e-6, 2e-6, 3e-6], dtype=np.float32),
               (1, 2): np.array([-4e-6, 0.0, 5e-6], dtype=np.float32)}
    _write_history(history, stamped)

    for _ in range(5):
        _rebuild(state, contacts, history)

    carried = history.as_map()
    assert set(carried) == set(stamped)
    for key, value in stamped.items():
        np.testing.assert_allclose(carried[key], value, rtol=1e-6)


def test_history_is_destroyed_when_the_contact_separates(device):
    """The whole point: a pair that parts must not resume where it left off."""
    state = _chain(device, 2, spacing=1.9 * RADIUS)
    contacts = ContactList(2, device)
    history = ContactHistory(2, device)
    _rebuild(state, contacts, history)
    _write_history(history, {(0, 1): np.array([9e-6, 0.0, 0.0], dtype=np.float32)})
    assert history.as_map(), "test would be vacuous with no history to destroy"

    # Move them apart, rebuild: the contact is gone.
    state.set_positions([[0.0, 0.0, 0.0], [4.0 * RADIUS, 0.0, 0.0]])
    _rebuild(state, contacts, history)
    assert history.as_map() == {}

    # Bring them back: the new contact must start from zero, not from 9e-6.
    state.set_positions([[0.0, 0.0, 0.0], [1.9 * RADIUS, 0.0, 0.0]])
    _rebuild(state, contacts, history)
    np.testing.assert_allclose(history.as_map()[(0, 1)], 0.0, atol=0.0)


def test_a_new_contact_starts_from_zero(device):
    """A freshly formed contact has not slipped yet, whatever occupied the slot."""
    state = _chain(device, 3, spacing=1.9 * RADIUS)
    contacts = ContactList(3, device)
    history = ContactHistory(3, device)
    _rebuild(state, contacts, history)
    _write_history(history, {
        (0, 1): np.array([7e-6, 0.0, 0.0], dtype=np.float32),
        (1, 2): np.array([8e-6, 0.0, 0.0], dtype=np.float32),
    })

    # Fold the chain into an equilateral triangle of side 1.9R, so 0-2 comes
    # into contact while 0-1 and 1-2 stay in contact throughout. Kept at 1.9R
    # rather than simply squeezing the line, which would need a separation
    # below 1.0R and an overlap of more than a full radius to close the 0-2
    # gap — physically absurd even in a test that computes no forces.
    state.set_positions([[0.0, 0.0, 0.0],
                         [1.9 * RADIUS, 0.0, 0.0],
                         [0.95 * RADIUS, 1.9 * RADIUS * np.sqrt(3.0) / 2.0, 0.0]])
    _rebuild(state, contacts, history)

    carried = history.as_map()
    assert set(carried) == {(0, 1), (0, 2), (1, 2)}
    np.testing.assert_allclose(carried[(0, 1)], [7e-6, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(carried[(1, 2)], [8e-6, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(carried[(0, 2)], 0.0, atol=0.0)


def test_history_follows_the_pair_not_the_slot(device):
    """Slot indices are nondeterministic; the pair->displacement map is not.

    Particle 0's neighbours are removed and restored so its slot occupancy
    changes between rebuilds. If history were keyed on slot index rather than
    on partner id, the displacements would end up attached to the wrong pairs.
    """
    n = 4
    state = ParticleState.allocate(n, device, RADIUS, DENSITY)
    close = [[0.0, 0.0, 0.0],
             [1.5 * RADIUS, 0.0, 0.0],
             [0.0, 1.5 * RADIUS, 0.0],
             [0.0, 0.0, 1.5 * RADIUS]]
    state.set_positions(close)
    contacts = ContactList(n, device)
    history = ContactHistory(n, device)
    _rebuild(state, contacts, history)

    stamped = {(0, 1): np.array([1e-6, 0.0, 0.0], dtype=np.float32),
               (0, 2): np.array([2e-6, 0.0, 0.0], dtype=np.float32),
               (0, 3): np.array([3e-6, 0.0, 0.0], dtype=np.float32)}
    _write_history(history, stamped)

    # Drop particle 2 out of contact, then bring it back. Slot order changes.
    moved = [row[:] for row in close]
    moved[2] = [0.0, 5.0 * RADIUS, 0.0]
    state.set_positions(moved)
    _rebuild(state, contacts, history)
    state.set_positions(close)
    _rebuild(state, contacts, history)

    carried = history.as_map()
    # 0-1 and 0-3 never broke, so they keep their values.
    np.testing.assert_allclose(carried[(0, 1)], stamped[(0, 1)], rtol=1e-6)
    np.testing.assert_allclose(carried[(0, 3)], stamped[(0, 3)], rtol=1e-6)
    # 0-2 broke and reformed, so it is a new contact.
    np.testing.assert_allclose(carried[(0, 2)], 0.0, atol=0.0)


def test_displacements_of_absent_slots_stay_zero(device):
    """Free slots must not accumulate junk that a later contact would inherit."""
    state = _chain(device, 3, spacing=1.9 * RADIUS)
    contacts = ContactList(3, device)
    history = ContactHistory(3, device)
    _rebuild(state, contacts, history)
    _write_history(history, {(0, 1): np.array([5e-6, 0.0, 0.0], dtype=np.float32)})
    _rebuild(state, contacts, history)

    partners = history.partners()
    displacements = history.displacements()
    empty = partners == EMPTY
    np.testing.assert_allclose(displacements[empty], 0.0, atol=0.0)


def test_rebuild_does_not_allocate(device):
    """Buffers are swapped by reference, so repeated rebuilds are allocation-free."""
    state = _chain(device, 3, spacing=1.9 * RADIUS)
    contacts = ContactList(3, device)
    history = ContactHistory(3, device)
    seen = set()
    for _ in range(4):
        _rebuild(state, contacts, history)
        seen.add(history.partner.ptr)
    # Exactly two buffers, alternating — never a third.
    assert len(seen) == 2
