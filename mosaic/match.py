"""Assign a pool image to each cell by colour, avoiding nearby repeats."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Layout


def match_cells(layout: Layout, targets: np.ndarray, pool_features: np.ndarray,
                *, no_repeat_radius: int = 3, candidates: int = 50):
    """Return (pool index per cell, number of cells that had to fall back).

    For each cell, take the closest of the ``candidates`` best colour
    matches that isn't already used by a cell within ``no_repeat_radius``
    grid steps (hex distance for hexes, Chebyshev distance for squares).
    If every candidate is taken, use the best match anyway.
    """
    n_pool = len(pool_features)
    if n_pool == 0:
        raise ValueError("empty pool")
    k = max(1, min(candidates, n_pool))
    _, idxs = cKDTree(pool_features).query(targets, k=k)
    idxs = np.asarray(idxs).reshape(len(targets), k)

    offsets = layout.neighbor_offsets(no_repeat_radius) if no_repeat_radius > 0 else []
    assigned = np.full(len(layout), -1, dtype=np.int64)
    fallbacks = 0
    for i, (q, r) in enumerate(layout.coords.tolist()):
        forbidden = set()
        for dq, dr in offsets:
            j = layout.index_of((q + dq, r + dr))
            if j is not None and assigned[j] >= 0:
                forbidden.add(int(assigned[j]))
        for cand in idxs[i]:
            if cand not in forbidden:
                assigned[i] = cand
                break
        else:
            assigned[i] = idxs[i, 0]
            fallbacks += 1
    return assigned, fallbacks
