"""Tile layouts: partitions of a W x H output canvas into cells.

Every layout answers the same questions, so matching and rendering don't
care whether the cells are rectangles or hexagons:

* ``centers``      (N, 2) float array of cell centres, ``[x, y]`` in pixels
* ``coords``       (N, 2) int array of grid coordinates (used for adjacency)
* ``cell_w/cell_h`` size of one cell's bounding box in pixels
* ``cell_index(x, y)``    which cell owns each point (-1 = none)
* ``edge_distance(dx, dy)`` signed distance from a point (relative to a cell
  centre) to that cell's boundary; positive inside
* ``neighbor_offsets(r)`` coordinate offsets of every cell within grid
  distance ``r``

Ownership is decided by ``cell_index`` alone, so cells tile the canvas
exactly: no gaps, no overlaps.
"""

from __future__ import annotations

import math

import numpy as np

SQRT3 = math.sqrt(3.0)


class Layout:
    name = "layout"

    width: int
    height: int
    cell_w: float
    cell_h: float
    centers: np.ndarray
    coords: np.ndarray

    def __len__(self) -> int:
        return len(self.centers)

    def _finish(self, centers, coords) -> None:
        self.centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        self.coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
        self._index = {tuple(c): i for i, c in enumerate(self.coords.tolist())}

    def index_of(self, coord) -> int | None:
        return self._index.get(tuple(coord))

    def template_mask(self, w: int, h: int) -> np.ndarray:
        """Boolean mask of one cell's shape stretched over a w x h raster."""
        xs = ((np.arange(w) + 0.5) / w - 0.5) * self.cell_w
        ys = ((np.arange(h) + 0.5) / h - 0.5) * self.cell_h
        dx, dy = np.meshgrid(xs, ys)
        return self.edge_distance(dx, dy) >= 0

    # Subclass API -------------------------------------------------------
    def cell_index(self, x, y) -> np.ndarray:
        raise NotImplementedError

    def edge_distance(self, dx, dy) -> np.ndarray:
        raise NotImplementedError

    def neighbor_offsets(self, radius: int) -> list[tuple[int, int]]:
        raise NotImplementedError


class RectLayout(Layout):
    """Axis-aligned grid of cell_w x cell_h rectangles anchored at (0, 0)."""

    name = "rect"

    def __init__(self, width: int, height: int, cell_w: float, cell_h: float):
        if cell_w <= 0 or cell_h <= 0:
            raise ValueError("cell size must be positive")
        self.width, self.height = int(width), int(height)
        self.cell_w, self.cell_h = float(cell_w), float(cell_h)
        self.cols = math.ceil(self.width / self.cell_w)
        self.rows = math.ceil(self.height / self.cell_h)
        rows, cols = np.mgrid[0:self.rows, 0:self.cols]
        centers = np.stack([(cols + 0.5) * self.cell_w,
                            (rows + 0.5) * self.cell_h], axis=-1)
        self._finish(centers, np.stack([cols, rows], axis=-1))

    def cell_index(self, x, y):
        col = np.floor(np.asarray(x) / self.cell_w).astype(np.int64)
        row = np.floor(np.asarray(y) / self.cell_h).astype(np.int64)
        ok = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        return np.where(ok, row * self.cols + col, -1)

    def edge_distance(self, dx, dy):
        return np.minimum(self.cell_w / 2 - np.abs(dx),
                          self.cell_h / 2 - np.abs(dy))

    def neighbor_offsets(self, radius):
        return [(dc, dr)
                for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if (dc, dr) != (0, 0)]


class HexLayout(Layout):
    """Regular hexagons of circumradius ``size`` in axial (q, r) coordinates.

    ``orientation='pointy'`` puts vertices at top and bottom (rows of hexes
    with vertical side edges); ``'flat'`` puts flat edges at top and bottom
    (columns of hexes).  Cell (0, 0) is centred on the canvas origin, and
    hexes overhanging the canvas edge are clipped by the renderer.
    """

    name = "hex"

    def __init__(self, width: int, height: int, size: float,
                 orientation: str = "pointy"):
        if size <= 0:
            raise ValueError("hex size must be positive")
        if orientation not in ("pointy", "flat"):
            raise ValueError("orientation must be 'pointy' or 'flat'")
        self.width, self.height = int(width), int(height)
        self.size = float(size)
        self.orientation = orientation
        self.name = f"hex-{orientation}"
        self.inner = self.size * SQRT3 / 2  # apothem

        if orientation == "pointy":
            self.cell_w, self.cell_h = SQRT3 * self.size, 2 * self.size
            angles = (0, 60, 120)
        else:
            self.cell_w, self.cell_h = 2 * self.size, SQRT3 * self.size
            angles = (30, 90, 150)
        rad = np.radians(angles)
        self._normals = np.stack([np.cos(rad), np.sin(rad)], axis=-1)

        # Axial bounding range of everything that could touch the canvas.
        mx, my = self.cell_w, self.cell_h
        cx = np.array([-mx, self.width + mx, -mx, self.width + mx])
        cy = np.array([-my, -my, self.height + my, self.height + my])
        qf, rf = self._pixel_to_axial(cx, cy)
        self._qmin, qmax = int(np.floor(qf.min())), int(np.ceil(qf.max()))
        self._rmin, rmax = int(np.floor(rf.min())), int(np.ceil(rf.max()))

        r, q = np.mgrid[self._rmin:rmax + 1, self._qmin:qmax + 1]
        q, r = q.ravel(), r.ravel()
        x, y = self._axial_to_pixel(q, r)
        keep = ((x > -self.cell_w / 2) & (x < self.width + self.cell_w / 2)
                & (y > -self.cell_h / 2) & (y < self.height + self.cell_h / 2))
        # Row-major order (top-to-bottom, then left-to-right) for readability.
        order = np.lexsort((x[keep], np.round(y[keep], 6)))
        q, r = q[keep][order], r[keep][order]
        x, y = x[keep][order], y[keep][order]
        self._finish(np.stack([x, y], axis=-1), np.stack([q, r], axis=-1))

        self._grid = np.full((rmax - self._rmin + 1, qmax - self._qmin + 1),
                             -1, dtype=np.int64)
        self._grid[r - self._rmin, q - self._qmin] = np.arange(len(q))

    def _axial_to_pixel(self, q, r):
        s = self.size
        if self.orientation == "pointy":
            return s * SQRT3 * (q + r / 2), s * 1.5 * r
        return s * 1.5 * q, s * SQRT3 * (r + q / 2)

    def _pixel_to_axial(self, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        s = self.size
        if self.orientation == "pointy":
            return (SQRT3 / 3 * x - y / 3) / s, (2 / 3 * y) / s
        return (2 / 3 * x) / s, (-x / 3 + SQRT3 / 3 * y) / s

    @staticmethod
    def _cube_round(qf, rf):
        sf = -qf - rf
        q, r, s = np.round(qf), np.round(rf), np.round(sf)
        dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
        fix_q = (dq > dr) & (dq > ds)
        fix_r = ~fix_q & (dr > ds)
        q = np.where(fix_q, -r - s, q)
        r = np.where(fix_r, -q - s, r)
        return q.astype(np.int64), r.astype(np.int64)

    def cell_index(self, x, y):
        q, r = self._cube_round(*self._pixel_to_axial(x, y))
        gi, gj = r - self._rmin, q - self._qmin
        ok = ((gi >= 0) & (gi < self._grid.shape[0])
              & (gj >= 0) & (gj < self._grid.shape[1]))
        out = np.full(np.shape(q), -1, dtype=np.int64)
        out[ok] = self._grid[gi[ok], gj[ok]]
        return out

    def edge_distance(self, dx, dy):
        dx, dy = np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64)
        proj = np.abs(dx[..., None] * self._normals[:, 0]
                      + dy[..., None] * self._normals[:, 1])
        return self.inner - proj.max(axis=-1)

    def neighbor_offsets(self, radius):
        return [(dq, dr)
                for dq in range(-radius, radius + 1)
                for dr in range(-radius, radius + 1)
                if (dq, dr) != (0, 0) and abs(dq + dr) <= radius]


def make_layout(shape: str, width: int, height: int, *, cols: int | None = None,
                rows: int | None = None, tile_size: float | None = None,
                orientation: str = "pointy") -> Layout:
    """Build a layout sized either by column count or by tile width (px).

    For hexes, ``cols`` counts hexes across the canvas and ``tile_size`` is
    the bounding-box width of one hex.  For squares, ``rows`` (optional)
    makes the cells rectangular, like the original fixed-grid script.
    """
    if shape == "hex":
        if tile_size is None:
            step = SQRT3 if orientation == "pointy" else 1.5  # x-spacing / size
            size = width / (cols * step)
        else:
            size = tile_size / (SQRT3 if orientation == "pointy" else 2.0)
        return HexLayout(width, height, size, orientation)
    if shape == "square":
        cell_w = tile_size if tile_size is not None else width / cols
        cell_h = height / rows if rows else cell_w
        return RectLayout(width, height, cell_w, cell_h)
    raise ValueError(f"unknown shape: {shape!r}")
