"""Assign a pool image to each cell by colour, avoiding nearby repeats.

Two matchers are available:

``greedy``
    Visit cells in reading order; take the closest colour match that isn't
    already used within the minimum distance.  Deterministic, but in large
    areas of uniform colour every cell sees the same ranked candidate list,
    so the assignment settles into a cycle that repeats as a visible
    lattice pattern.

``random``
    Visit cells in a seeded random order, and treat every allowed candidate
    whose colour is within ``tolerance`` (CIELAB delta-E) of the best one as
    equally good, choosing among them at random.  Uniform areas get an
    irregular scatter of near-identical-colour images instead of a cycle.

Both accept ``reuse_penalty``: each prior use of an image adds that many
delta-E units to its cost, spreading usage across the pool.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Layout

MATCHERS = ("random", "greedy")


def match_cells(layout: Layout, targets: np.ndarray, pool_features: np.ndarray, *,
                min_distance: int = 4, candidates: int = 100, matcher: str = "random",
                tolerance: float = 6.0, reuse_penalty: float = 0.0, seed: int = 0):
    """Return (pool index per cell, number of cells that had to break the rule).

    ``min_distance``: two uses of the same image are at least this many
    cells apart (hex steps for hexes, king-move steps for squares); 0 or 1
    disables the rule.  ``candidates``: best colour matches considered per
    cell.  If all of them are excluded, the cheapest is used anyway and
    counted as a fallback.
    """
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; choose from {MATCHERS}")
    n_pool = len(pool_features)
    if n_pool == 0:
        raise ValueError("empty pool")
    n = len(layout)
    k = max(1, min(candidates, n_pool))
    dists, idxs = cKDTree(pool_features).query(targets, k=k)
    dists = np.asarray(dists, dtype=np.float64).reshape(n, k)
    idxs = np.asarray(idxs).reshape(n, k)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n) if matcher == "random" else range(n)
    radius = max(0, min_distance - 1)
    offsets = layout.neighbor_offsets(radius) if radius > 0 else []
    coords = layout.coords.tolist()

    assigned = np.full(n, -1, dtype=np.int64)
    uses = np.zeros(n_pool)
    fallbacks = 0
    for i in order:
        q, r = coords[i]
        forbidden = []
        for dq, dr in offsets:
            j = layout.index_of((q + dq, r + dr))
            if j is not None and assigned[j] >= 0:
                forbidden.append(assigned[j])
        cost = dists[i] + reuse_penalty * uses[idxs[i]] if reuse_penalty else dists[i].copy()
        if forbidden:
            blocked = np.isin(idxs[i], forbidden)
            if blocked.all():
                fallbacks += 1
            else:
                cost[blocked] = np.inf
        if matcher == "random":
            near = np.flatnonzero(cost <= cost.min() + tolerance)
            pick = near[rng.integers(len(near))]
        else:
            pick = int(np.argmin(cost))
        assigned[i] = idxs[i, pick]
        uses[assigned[i]] += 1
    return assigned, fallbacks


def repetition_score(layout: Layout, assigned: np.ndarray, max_offset: int = 12) -> float:
    """Largest fraction of cells whose image recurs at one fixed grid offset.

    A periodic (patterned) assignment scores high for its period; an
    irregular one stays near zero.  Used for diagnostics and tests.
    """
    best = 0.0
    coords = layout.coords.tolist()
    for dq in range(-max_offset, max_offset + 1):
        for dr in range(0, max_offset + 1):
            if (dr == 0 and dq <= 0):
                continue
            same = total = 0
            for i, (q, r) in enumerate(coords):
                j = layout.index_of((q + dq, r + dr))
                if j is not None:
                    total += 1
                    same += assigned[i] == assigned[j]
            if total:
                best = max(best, same / total)
    return best
