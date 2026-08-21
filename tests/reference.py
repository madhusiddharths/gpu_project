"""Host reference implementations. Deliberately slow and obviously correct.

These are the oracle for the oracle: the Warp O(N^2) kernel is checked against
these, and from Phase 2 the grid is checked against the Warp kernel.

Written in NumPy with full pairwise matrices, so memory is O(N^2) and these are
usable only at test scale. That is intentional — resisting the urge to optimise
a reference implementation is what keeps it trustworthy.
"""

import numpy as np


def naive_pair_set(positions: np.ndarray, radii: np.ndarray) -> set[tuple[int, int]]:
    """Every unordered pair (i, j) with i < j whose spheres strictly overlap."""
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(positions)

    delta = positions[:, None, :] - positions[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", delta, delta)
    cutoff = radii[:, None] + radii[None, :]

    overlapping = dist_sq < cutoff**2
    rows, cols = np.triu_indices(n, k=1)
    selected = overlapping[rows, cols]
    return set(zip(rows[selected].tolist(), cols[selected].tolist(), strict=True))