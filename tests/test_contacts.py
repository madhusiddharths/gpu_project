"""Contact detection tests.

Comparisons are against SETS of pairs, never arrays. Slots in the pair list are
claimed with an atomic counter, so ordering depends on thread scheduling and
differs between runs and between devices. Contents are deterministic; order is
not. Asserting on order would produce a test that passes on CPU and fails on
CUDA for no physical reason.
"""

import numpy as np
import pytest
import warp as wp
from reference import naive_pair_set

from warp_dem import resolve_device
from warp_dem.contacts import ContactList, ContactOverflow
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


def _place(device, positions, radii=None):
    positions = np.asarray(positions, dtype=np.float32)
    n = len(positions)
    state = ParticleState.allocate(n, device, RADIUS, DENSITY)
    state.set_positions(positions)
    if radii is not None:
        state.radius.assign(np.asarray(radii, dtype=np.float32))
    return state


def test_two_overlapping_particles_are_one_pair(device):
    state = _place(device, [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    contacts = ContactList(2, device)
    assert contacts.detect(state) == 1
    assert contacts.pair_set() == {(0, 1)}


def test_two_separated_particles_are_no_pair(device):
    state = _place(device, [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    contacts = ContactList(2, device)
    assert contacts.detect(state) == 0
    assert contacts.pair_set() == set()


def test_exactly_touching_is_not_a_contact(device):
    """Strict inequality. Zero overlap is zero force, so the pair would occupy
    a slot and a tangential-history entry while contributing nothing."""
    state = _place(device, [[0.0, 0.0, 0.0], [2.0 * RADIUS, 0.0, 0.0]])
    contacts = ContactList(2, device)
    assert contacts.detect(state) == 0


def test_just_inside_the_boundary_is_a_contact(device):
    """The companion to the test above: the cutoff is in the right place, not
    merely consistently strict."""
    state = _place(device, [[0.0, 0.0, 0.0], [2.0 * RADIUS * 0.9999, 0.0, 0.0]])
    contacts = ContactList(2, device)
    assert contacts.detect(state) == 1


def test_no_self_pairs_and_no_duplicates(device):
    """A dense clump, where every particle touches every other."""
    rng = np.random.default_rng(1)
    n = 12
    positions = rng.uniform(-RADIUS, RADIUS, size=(n, 3))

    state = _place(device, positions)
    contacts = ContactList(n, device)
    found = contacts.detect(state)
    pairs = contacts.pairs()

    assert len(pairs) == found
    assert len(contacts.pair_set()) == found, "duplicate pair in the list"
    assert np.all(pairs[:, 0] < pairs[:, 1]), "pair not stored in i<j order"


def test_matches_numpy_reference_on_a_random_cloud(device):
    """The Block 9 done-when criterion."""
    rng = np.random.default_rng(0)
    n = 800
    positions = rng.uniform(0.0, 0.03, size=(n, 3))

    state = _place(device, positions)
    contacts = ContactList(n, device)
    found = contacts.detect(state)

    expected = naive_pair_set(positions, np.full(n, RADIUS))
    assert found == len(expected)
    assert contacts.pair_set() == expected
    assert found > 50, "test cloud is too dilute to exercise anything"


def test_matches_reference_with_polydisperse_radii(device):
    """A detector using only radius[i] instead of radius[i] + radius[j] passes
    the monodisperse test and fails here.

    Box size is set so the solid fraction is ~0.57 rather than pathological: at
    20 mm these radii give a solid fraction of 4.5 — the spheres' combined
    volume exceeds the container's — and every particle contacts 22 others,
    which correctly overflows the list. Capacity pressure is tested separately;
    this test is about the cutoff.
    """
    rng = np.random.default_rng(7)
    n = 400
    positions = rng.uniform(0.0, 0.04, size=(n, 3))
    radii = rng.uniform(0.5 * RADIUS, 2.0 * RADIUS, size=n)

    state = _place(device, positions, radii)
    contacts = ContactList(n, device)
    found = contacts.detect(state)

    expected = naive_pair_set(positions, radii)
    assert contacts.pair_set() == expected
    assert found > 100, "cloud too dilute to distinguish the cutoff rules"
    

def test_detection_is_repeatable(device):
    """Contents must be stable across repeated launches even though slot order
    is not. This is the invariant that survives the move to CUDA."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(0.0, 0.02, size=(300, 3))

    state = _place(device, positions)
    contacts = ContactList(300, device)

    contacts.detect(state)
    first = contacts.pair_set()
    for _ in range(3):
        contacts.detect(state)
        assert contacts.pair_set() == first


def test_list_is_cleared_between_detections(device):
    """A stale counter would make the second detection append to the first."""
    dense = _place(device, np.random.default_rng(5).uniform(-RADIUS, RADIUS, (10, 3)))
    contacts = ContactList(10, device)
    busy = contacts.detect(dense)
    assert busy > 0

    dense.set_positions(np.arange(10)[:, None] * np.array([[0.01, 0.0, 0.0]]))
    assert contacts.detect(dense) == 0
    assert contacts.pair_set() == set()


def test_overflow_raises_and_reports_the_shortfall(device):
    rng = np.random.default_rng(11)
    n = 60
    positions = rng.uniform(-RADIUS, RADIUS, size=(n, 3))

    state = _place(device, positions)
    contacts = ContactList(n, device, capacity=8)

    with pytest.raises(ContactOverflow) as exc:
        contacts.detect(state)

    message = str(exc.value)
    assert "capacity is 8" in message
    assert "contacts per particle" in message


def test_exact_capacity_is_sufficient(device):
    """Off-by-one guard: a list holding precisely the number of contacts found
    must not report overflow."""
    positions = [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.0, 0.003, 0.0]]
    state = _place(device, positions)

    counted = ContactList(3, device).detect(state)
    tight = ContactList(3, device, capacity=counted)
    assert tight.detect(state) == counted
    assert len(tight.pair_set()) == counted


def test_size_mismatch_is_rejected(device):
    state = _place(device, [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]])
    with pytest.raises(ValueError):
        ContactList(5, device).detect(state)