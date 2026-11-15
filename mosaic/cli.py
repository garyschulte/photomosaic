"""Command-line interface for the photomosaic generator."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from .geometry import make_layout
from .match import MATCHERS, match_cells
from .pool import build_pool
from .render import load_source, render, sample_targets


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="create_mosaic.py",
        description="Build a photomosaic from a source image and a corpus of photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io_ = p.add_argument_group("input / output")
    io_.add_argument("-s", "--source", required=True, help="source image to reproduce")
    io_.add_argument("-c", "--corpus", default="corpus", help="directory of tile images (searched recursively)")
    io_.add_argument("-o", "--out", default="mosaic.png", help="output image (PNG recommended)")

    tiles = p.add_argument_group("tiles")
    tiles.add_argument("--shape", choices=["hex", "square"], default="hex")
    tiles.add_argument("--orientation", choices=["pointy", "flat"], default="pointy",
                       help="hex orientation: 'pointy' = vertex on top, 'flat' = edge on top")
    size = tiles.add_mutually_exclusive_group()
    size.add_argument("--cols", type=int, default=100, help="number of tiles across")
    size.add_argument("--tile-size", type=float,
                      help="tile width in output pixels (overrides --cols)")
    tiles.add_argument("--rows", type=int,
                       help="square shape only: fixed number of rows (cells become rectangles)")

    out = p.add_argument_group("output size")
    osz = out.add_mutually_exclusive_group()
    osz.add_argument("--scale", type=float, default=1.0,
                     help="output size relative to the source image")
    osz.add_argument("--width", type=int, help="output width in pixels (height keeps aspect)")

    look = p.add_argument_group("matching / appearance")
    look.add_argument("--matcher", choices=MATCHERS, default="random",
                      help="'random': shuffled visit order + random choice among near-equal "
                           "colours (avoids repeating patterns in uniform areas); "
                           "'greedy': reading order, always the closest colour")
    look.add_argument("--min-distance", type=int, default=4, metavar="N",
                      help="two uses of the same image are at least N cells apart "
                           "(0 or 1 = no limit)")
    look.add_argument("--tolerance", type=float, default=6.0, metavar="DE",
                      help="random matcher: candidates within this CIELAB delta-E of the "
                           "best match count as equally good (~2.3 is just noticeable)")
    look.add_argument("--reuse-penalty", type=float, default=0.0, metavar="DE",
                      help="delta-E added to an image's cost per prior use; spreads usage "
                           "across the pool")
    look.add_argument("--candidates", type=int, default=100,
                      help="best colour matches considered per cell")
    look.add_argument("--blend", type=float, default=0.0,
                      help="0..1: shift each tile's colour toward the source colour")
    look.add_argument("--grout", type=float, default=0.0,
                      help="width in px of lines drawn between tiles")
    look.add_argument("--grout-color", default="#000000", help="grout / background colour")

    perf = p.add_argument_group("pool / performance")
    perf.add_argument("--max-pool", type=int, help="use a random subset of N corpus images")
    perf.add_argument("--seed", type=int, default=42, help="seed for --max-pool subsetting and the random matcher")
    perf.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    perf.add_argument("--cache-dir", default=".mosaic_cache",
                      help="where to cache corpus colour features")
    perf.add_argument("--no-cache", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows and args.shape != "square":
        print("error: --rows only applies to --shape square", file=sys.stderr)
        return 2
    if not 0 <= args.blend <= 1:
        print("error: --blend must be in [0, 1]", file=sys.stderr)
        return 2
    if not os.path.isfile(args.source):
        print(f"error: source image not found: {args.source}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.corpus):
        print(f"error: corpus directory not found: {args.corpus}", file=sys.stderr)
        return 1

    t0 = time.time()
    with Image.open(args.source) as im:
        src_w, src_h = ImageOps.exif_transpose(im).size
    if args.width:
        W = args.width
    else:
        W = round(src_w * args.scale)
    H = round(W * src_h / src_w)

    layout = make_layout(args.shape, W, H, cols=args.cols, rows=args.rows,
                         tile_size=args.tile_size, orientation=args.orientation)
    print(f"Source {src_w}x{src_h} -> output {W}x{H}; {len(layout)} {layout.name} cells "
          f"({layout.cell_w:.1f}x{layout.cell_h:.1f} px)")
    if min(layout.cell_w, layout.cell_h) < 8:
        print("  warning: tiles are under 8 px; consider fewer --cols or a larger --scale")

    print("Building pool ...")
    pool = build_pool(args.corpus, layout,
                      cache_dir=None if args.no_cache else args.cache_dir,
                      workers=args.workers, max_pool=args.max_pool, seed=args.seed)
    if len(pool) == 0:
        print("error: no usable images in corpus", file=sys.stderr)
        return 1
    print(f"  Pool ready: {len(pool)} images")

    print("Sampling source colours ...")
    # Enough resolution for ~12 samples across each cell, capped for memory.
    cols_across = W / layout.cell_w
    src, _ = load_source(args.source, max_side=int(min(4000, max(1600, 12 * cols_across))))
    target_lab, target_rgb = sample_targets(layout, src)

    print("Matching ...")
    assignment, fallbacks = match_cells(layout, target_lab, pool.features,
                                        matcher=args.matcher, min_distance=args.min_distance,
                                        tolerance=args.tolerance,
                                        reuse_penalty=args.reuse_penalty,
                                        candidates=args.candidates, seed=args.seed)
    counts = np.bincount(assignment)
    print(f"  {args.matcher} matcher: {np.count_nonzero(counts)} distinct images, "
          f"most-used image appears {counts.max()}x"
          + (f"; {fallbacks} cells broke --min-distance (raise --candidates)"
             if fallbacks else ""))

    print("Rendering ...")
    canvas = render(layout, assignment, pool.paths, target_rgb,
                    blend=args.blend, grout=args.grout, grout_color=args.grout_color,
                    workers=args.workers)
    Image.fromarray(canvas).save(args.out)
    print(f"Mosaic saved to {args.out}  ({time.time() - t0:.1f}s)")
    return 0
