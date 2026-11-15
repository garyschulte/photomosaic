"""Source sampling and mosaic rendering."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps

from .color import srgb_to_lab
from .geometry import Layout
from .pool import load_tile


def load_source(path: str, max_side: int | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (float RGB array for colour sampling, original (w, h)).

    The sampling copy is downscaled to at most *max_side* px on its long
    side; only cell averages are taken from it, so full resolution is
    unnecessary.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        size = im.size
        if max_side and max(size) > max_side:
            im = im.copy()
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0, size


def sample_targets(layout: Layout, src: np.ndarray):
    """Mean (Lab, RGB) colour of the source under each cell.

    *src* may be any resolution; it is mapped onto the layout's canvas.
    """
    sh, sw = src.shape[:2]
    xs = (np.arange(sw) + 0.5) * (layout.width / sw)
    ys = (np.arange(sh) + 0.5) * (layout.height / sh)
    labels = layout.cell_index(*np.meshgrid(xs, ys)).ravel()
    ok = labels >= 0
    n = len(layout)
    counts = np.bincount(labels[ok], minlength=n).astype(np.float64)

    flat_rgb = src.reshape(-1, 3)[ok]
    flat_lab = srgb_to_lab(src).reshape(-1, 3)[ok]
    out = []
    for flat in (flat_lab, flat_rgb):
        sums = np.stack([np.bincount(labels[ok], flat[:, c], minlength=n)
                         for c in range(3)], axis=-1)
        out.append(sums / np.maximum(counts, 1)[:, None])

    # Cells too small to contain a sample pixel (edge slivers): use the
    # nearest source pixel to the cell centre.
    empty = counts == 0
    if empty.any():
        cx = np.clip((layout.centers[empty, 0] * sw / layout.width).astype(int), 0, sw - 1)
        cy = np.clip((layout.centers[empty, 1] * sh / layout.height).astype(int), 0, sh - 1)
        out[0][empty] = srgb_to_lab(src[cy, cx])
        out[1][empty] = src[cy, cx]
    return out[0], out[1]


def parse_color(text: str) -> np.ndarray:
    text = text.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected #rrggbb, got {text!r}")
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def render(layout: Layout, assignment: np.ndarray, paths: list[str],
           target_rgb: np.ndarray, *, blend: float = 0.0, grout: float = 0.0,
           grout_color: str = "#000000", workers: int = 8, log=print) -> np.ndarray:
    """Paint each cell with its matched image, clipped to the cell shape.

    ``blend`` in [0, 1] shifts each tile's mean colour toward the source
    colour of its cell.  ``grout`` draws lines of that width (px) between
    cells.
    """
    W, H = layout.width, layout.height
    gcolor = parse_color(grout_color)
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = gcolor.astype(np.uint8)

    # Tile raster: the cell's bounding box plus a pixel of slack per side.
    tw, th = math.ceil(layout.cell_w) + 2, math.ceil(layout.cell_h) + 2

    unique = sorted(set(int(a) for a in assignment))
    log(f"  Loading {len(unique)} distinct tile images at {tw}x{th} px ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        tiles = dict(zip(unique, ex.map(lambda i: load_tile(paths[i], (tw, th)), unique)))

    half_grout = grout / 2
    for i, (cx, cy) in enumerate(layout.centers):
        x0, y0 = math.floor(cx - tw / 2), math.floor(cy - th / 2)
        # Clip the tile box to the canvas.
        ax0, ay0 = max(x0, 0), max(y0, 0)
        ax1, ay1 = min(x0 + tw, W), min(y0 + th, H)
        if ax0 >= ax1 or ay0 >= ay1:
            continue
        px = np.arange(ax0, ax1) + 0.5
        py = np.arange(ay0, ay1) + 0.5
        X, Y = np.meshgrid(px, py)
        own = layout.cell_index(X, Y) == i
        if not own.any():
            continue
        tile = tiles[int(assignment[i])][ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        tile = tile.astype(np.float64)
        if blend > 0:
            shift = target_rgb[i] * 255.0 - tile[own].mean(axis=0)
            tile = np.clip(tile + blend * shift, 0, 255)
        if half_grout > 0:
            edge = layout.edge_distance(X - cx, Y - cy) < half_grout
            tile[edge] = gcolor
        region = canvas[ay0:ay1, ax0:ax1]
        region[own] = np.rint(tile[own]).astype(np.uint8)
    return canvas
