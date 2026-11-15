#!/usr/bin/env python3
"""
Photo Mosaic Creator — Proof of Concept

Uses the photomosaic package (by Daniel Ballan) to create a mosaic from:
  - A source image (photomosaic/source_1.jpg)
  - A corpus of tile images (photomosaic/corpus/**)

Output: photomosaic/mosaic_1.png (lossless PNG — avoids JPEG artifacts
       when saving the fine tile detail)

To regenerate:   python3 create_mosaic.py

To adjust: edit the CONFIG section below, then re-run.

NOTE: This script patches two bugs in the photomospace library:
      1) np.product was removed in NumPy 1.25+; we alias it to np.prod.
      2) crop_to_fit uses np.floor/np.ceil producing float slice indices,
         which skimage.util.crop rejects; we cast them to int.
"""

import os
import sys
import time
import glob

import numpy as np
import photomosaic as pm
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Workarounds for photomosaic library bugs
# ---------------------------------------------------------------------------

# np.product was removed in NumPy 1.25+.  Patch it back for photomosaic.
if not hasattr(np, 'product'):
    np.product = np.prod


# Patch photomosaic.crop_to_fit to fix float slice indices.
# The library uses np.floor/np.ceil producing floats passed to
# skimage.util.crop (which requires int slices).  Our patch casts them.
pm_inner = pm.photomosaic
_orig_resize = pm_inner.resize

if hasattr(pm_inner, 'crop_to_fit'):
    def patched_crop_to_fit(image, shape):
        d = np.argmin(np.array(image.shape)[:2] / np.array(shape))
        enlarged_shape = (
            tuple(int(np.ceil(image.shape[i] * shape[d] / image.shape[d]))
                  for i in range(len(shape)))
            + image.shape[len(shape):]
        )
        resized = _orig_resize(
            image, enlarged_shape, mode='constant', anti_aliasing=False)
        crop_width = []
        for actual, target in zip(resized.shape, shape):
            overflow = int(actual) - int(target)
            left_margin = int(np.floor(overflow / 2))
            right_margin = int(np.ceil(overflow / 2))
            crop_width.append(slice(left_margin, actual - right_margin))
        for _ in range(resized.ndim - len(shape)):
            crop_width.append(
                slice(0, resized.shape[-(resized.ndim - len(shape))])
                if resized.ndim > len(shape) else slice(0, resized.shape[-1]))
        return resized[tuple(crop_width)]

    pm_inner.crop_to_fit = patched_crop_to_fit

# Also patch rescale_commensurate (used by the library for depth > 0).
if hasattr(pm, 'rescale_commensurate'):
    def patched_rescale(image, grid_dims, depth=0):
        factor = np.array(grid_dims) * 2**depth
        new_shape = [
            int(factor[i] * np.ceil(image.shape[i] / factor[i]))
            for i in [0, 1]]
        d = np.argmin(np.array(image.shape)[:2] / np.array(new_shape))
        enlarged_shape = (
            tuple(int(np.ceil(image.shape[i] * new_shape[d] / image.shape[d]))
                  for i in range(len(new_shape)))
            + image.shape[2:])
        resized = resize(image, enlarged_shape,
                         mode='constant', anti_aliasing=False)
        crop_width = []
        for actual, target in zip(resized.shape, new_shape):
            overflow = int(actual) - int(target)
            left_margin = int(np.floor(overflow / 2))
            right_margin = int(np.ceil(overflow / 2))
            crop_width.append(slice(left_margin, actual - right_margin))
        for _ in range(resized.ndim - len(new_shape)):
            crop_width.append(
                slice(0, resized.shape[-(resized.ndim - len(new_shape))])
                if resized.ndim > len(new_shape) else slice(0, resized.shape[-1]))
        return resized[tuple(crop_width)]

    # Patch BOTH the inner module (for draw_mosaic's closure) and top-level pm
    pm_inner.rescale_commensurate = patched_rescale
    pm.rescale_commensurate = patched_rescale


# ---------------------------------------------------------------------------
# CONFIG — tweak these to customize the mosaic
# ---------------------------------------------------------------------------

SOURCE_IMAGE = os.path.join("photomosaic", "source_1.jpg")
CORPUS_DIR = os.path.join("photomosaic", "corpus")
OUTPUT_IMAGE = os.path.join("photomosaic", "mosaic_1.png")

GRID_HEIGHT = 50      # rows of tiles
GRID_WIDTH = 100      # columns of tiles
DEPTH = 0             # 0 = uniform grid; 1 = adaptive subdivision
SCALE = 2             # 1 = 1x original size, 2 = 2x, etc.
MAX_POOL_SIZE = None  # None = use all images; set to e.g. 200 to test fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_pool(corpus_dir):
    """
    Scan *corpus_dir* for .jpg / .JPG files and build a photomosaic pool.
    Mislabeled mp4 files (saved as .jpg) are silently skipped.
    """
    jpg_files = (
        glob.glob(os.path.join(corpus_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(corpus_dir, "**", "*.JPG"), recursive=True))

    # Exclude mislabeled mp4 files
    jpg_files = [f for f in jpg_files
                 if "mp.jpg" not in f.lower() and "mp4.jpg" not in f.lower()]

    print(f"  Found {len(jpg_files)} image files in corpus.")

    # Merge .jpg and .JPG pools
    pool = pm.make_pool(os.path.join(corpus_dir, "**", "*.jpg"))
    pool2 = pm.make_pool(os.path.join(corpus_dir, "**", "*.JPG"))
    for k, v in pool2.items():
        if k not in pool:
            pool[k] = v

    if MAX_POOL_SIZE and len(pool) > MAX_POOL_SIZE:
        keys = list(pool.keys())
        np.random.seed(42)
        subset = np.random.choice(keys, MAX_POOL_SIZE, replace=False)
        pool = {k: pool[k] for k in subset}
        print(f"  (truncated pool to {MAX_POOL_SIZE} images)")

    print(f"  Pool ready: {len(pool)} valid tile images.")
    return pool


def preprocess_image(image_path):
    """
    Load, convert to perceptual colour space, and rescale/crop so the
    dimensions are evenly divisible by GRID_HEIGHT x GRID_WIDTH.
    """
    image = io.imread(image_path)
    print(f"Source image: {image.shape[1]}x{image.shape[0]}  ({image.dtype})")
    image = img_as_float(image)
    converted = pm.perceptual(image)

    # Compute grid-compatible target size
    new_h = int(GRID_HEIGHT * np.ceil(converted.shape[0] / GRID_HEIGHT))
    new_w = int(GRID_WIDTH * np.ceil(converted.shape[1] / GRID_WIDTH))
    processed = pm.rescale_commensurate(converted, (GRID_HEIGHT, GRID_WIDTH),
                                       depth=DEPTH)
    print(f"  Resized + cropped to: {processed.shape[1]}x{processed.shape[0]}")
    return processed


def _compute_tile_colors(tiles, processed_image):
    """Compute the average colour of each tile."""
    return [
        np.mean(processed_image[tile].reshape(-1, 3), axis=0)
        for tile in tiles]


def match_tiles_no_repeat(tiles, tile_colors, pool,
                          grid_h, grid_w, max_neighbors=3):
    """
    Match each tile to the closest pool image, **avoiding reuse** of the
    same pool image within *max_neighbors* slots in any direction
    (horizontal, vertical, diagonal).

    The library's simple_matcher picks the single best match for each
    tile independently, which frequently reuses the same pool image for
    adjacent slots when colours are similar.  Instead we solve this as
    a nearest-neighbour search with a "forbidden" set:

    1. Build a cKDTree over pool colours for fast k-nearest search.
    2. For each tile, query the k best matches (k = pool size or
       enough to guarantee a usable choice).
    3. Skip any match that was assigned to a neighbour within
       *max_neighbors* tiles (Manhattan distance <= max_neighbors).
    4. Fall back to the best match if no restricted choice is available.

    Returns a list of matched pool keys (one per tile).
    """
    pool_keys = list(pool.keys())
    pool_vecs = np.array([pool[k] for k in pool_keys])

    # cKDTree for fast nearest-neighbour queries
    tree = pm.cKDTree(pool_vecs)
    # Query the top k matches for every tile (k = pool size for safety)
    k = min(len(pool_keys), 50)  # 50 best candidates per tile
    dists, idxs = tree.query(np.array(tile_colors), k=k)

    # We track which pool index has been used at which tile index.
    # A pool key is "forbidden" for tile i if it was assigned to any
    # tile j where the Manhattan distance |row_i - row_j| + |col_i - col_j|
    # <= max_neighbors.
    assigned = [None] * len(tiles)  # pool index assigned to each tile

    for i in range(len(tiles)):
        # Get the tile slice bounds for neighbour-checking
        t = tiles[i]
        h_sl = t[0] if isinstance(t, (tuple, list)) else t
        w_sl = t[1] if isinstance(t, (tuple, list)) else slice(None)
        # Use the last column as the column coordinate when stop is None
        ri = int(h_sl.stop) if h_sl.stop is not None else GRID_HEIGHT
        ci = int(w_sl.stop) if (w_sl.stop is not None and w_sl.stop != GRID_WIDTH) else GRID_WIDTH
        # Iterate over all already-assigned tiles
        forbidden_indices = set()
        for j in range(i):
            if assigned[j] is None:
                continue
            jt = tiles[j]
            jh_sl = jt[0] if isinstance(jt, (tuple, list)) else jt
            jw_sl = jt[1] if isinstance(jt, (tuple, list)) else slice(None)
            rj = int(jh_sl.stop) if jh_sl.stop is not None else GRID_HEIGHT
            cj = int(jw_sl.stop) if (jw_sl.stop is not None and jw_sl.stop != GRID_WIDTH) else GRID_WIDTH
            if abs(ri - rj) <= max_neighbors and abs(ci - cj) <= max_neighbors:
                forbidden_indices.add(assigned[j])

        # Find the best candidate (closest to this tile's colour)
        # that is not forbidden.
        best_pool_idx = None
        for ci2 in range(k):
            if idxs[i, ci2] not in forbidden_indices:
                best_pool_idx = idxs[i, ci2]
                break

        if best_pool_idx is None:
            # All k candidates forbidden — just take the best match
            best_pool_idx = idxs[i, 0]

        assigned[i] = best_pool_idx

    return [pool_keys[pi] for pi in assigned]


def match_tiles(tiles, processed_image, pool):
    """
    Compute average colour of each tile and match to closest pool image,
    avoiding reuse of the same tile within 3 slots in any direction.
    """
    tile_colors = _compute_tile_colors(tiles, processed_image)
    matches = match_tiles_no_repeat(
        tiles, tile_colors, pool,
        grid_h=GRID_HEIGHT, grid_w=GRID_WIDTH, max_neighbors=3)
    print(f"  Matched {len(matches)} tiles to pool images "
          f"(no-repeat within 3 cells).")
    return matches


def draw_mosaic_result(tiles, matches, grid_shape):
    """
    Assemble the final mosaic.

    NOTE: pm.draw_mosaic already returns RGB (float64, 0-1).  Calling
    pm.rgb() on top of it re-converts RGB -> CIECAM02 -> RGB and fails for
    dark pixels (divide by zero), turning them to NaN -> 0 (black).  We
    return draw_mosaic's output directly.
    """
    canvas = np.ones_like(np.zeros((grid_shape[0], grid_shape[1], 3)))
    cache = {}
    mosaic = pm.draw_mosaic(
        canvas, tiles, matches,
        scale=SCALE,
        resized_copy_cache=cache)
    # Return directly — pm.draw_mosaic already returns RGB
    return mosaic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    if not os.path.isfile(SOURCE_IMAGE):
        print(f"Error: source image not found: {SOURCE_IMAGE}")
        sys.exit(1)
    if not os.path.isdir(CORPUS_DIR):
        print(f"Error: corpus directory not found: {CORPUS_DIR}")
        sys.exit(1)

    # 1. Build pool
    pool = build_pool(CORPUS_DIR)
    if len(pool) == 0:
        print("Error: no valid images found in corpus.")
        sys.exit(1)

    # 2. Preprocess source
    processed_image = preprocess_image(SOURCE_IMAGE)

    # 3. Partition into tiles
    #    Use the library's rescale_commensurate (patched) and partition
    processed_image = pm.rescale_commensurate(
        processed_image, (GRID_HEIGHT, GRID_WIDTH), depth=DEPTH)
    tiles = pm.partition(processed_image, (GRID_HEIGHT, GRID_WIDTH),
                         depth=DEPTH)
    print(f"  Partitioned into {len(tiles)} tiles (grid={GRID_HEIGHT}x{GRID_WIDTH}).")

    # 4. Match tiles to pool
    matches = match_tiles(tiles, processed_image, pool)

    # 5. Draw mosaic (pm.draw_mosaic returns RGB directly)
    print("Drawing mosaic (this may take a while) ...")
    result = draw_mosaic_result(tiles, matches, processed_image.shape[:2])
    result = np.clip(result, 0, 1)

    # 6. Save
    io.imsave(OUTPUT_IMAGE, (result * 255).astype(np.uint8))
    elapsed = time.time() - t0
    print(f"Mosaic saved to {OUTPUT_IMAGE}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
