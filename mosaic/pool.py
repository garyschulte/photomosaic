"""Corpus scanning, tile loading, and the per-image colour features.

Each corpus image's feature is its mean CIELAB colour *under the tile
shape*: the image is centre-cropped to the cell's aspect ratio exactly as
the renderer will crop it, then averaged inside the hex (or rectangle)
mask.  Features are cached on disk, keyed by path + mtime + size.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from .color import srgb_to_lab
from .geometry import Layout

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CACHE_VERSION = 1
FEATURE_HEIGHT = 32  # thumbnail height used for colour features


@dataclass
class Pool:
    paths: list[str]
    features: np.ndarray  # (N, 3) mean Lab

    def __len__(self) -> int:
        return len(self.paths)


def scan_corpus(corpus_dir: str) -> list[str]:
    found = []
    for root, dirs, files in os.walk(corpus_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if (not name.startswith(".")
                    and os.path.splitext(name)[1].lower() in IMAGE_EXTS):
                found.append(os.path.join(root, name))
    return sorted(found)


def load_tile(path: str, size: tuple[int, int]) -> np.ndarray:
    """Load *path* as a uint8 (h, w, 3) array, centre-cropped to *size*.

    Honours EXIF orientation (phone photos) and uses JPEG draft mode so
    large files decode at a reduced scale.
    """
    w, h = size
    with Image.open(path) as im:
        side = max(w, h)
        im.draft("RGB", (side, side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(ImageOps.fit(im, (w, h), Image.Resampling.LANCZOS))


def _feature(path, size, mask):
    try:
        lab = srgb_to_lab(load_tile(path, size) / 255.0)
    except Exception:  # unreadable / not really an image
        return None
    return lab[mask].mean(axis=0)


def _cache_file(cache_dir, corpus_dir, layout, size):
    key = f"{CACHE_VERSION}|{os.path.abspath(corpus_dir)}|{layout.name}|{size}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"pool-{digest}.npz")


def build_pool(corpus_dir: str, layout: Layout, *, cache_dir: str | None,
               workers: int = 8, max_pool: int | None = None,
               seed: int = 0, log=print) -> Pool:
    paths = scan_corpus(corpus_dir)
    log(f"  Found {len(paths)} image files in {corpus_dir}")
    if not paths:
        return Pool([], np.zeros((0, 3)))

    th = FEATURE_HEIGHT
    tw = max(1, round(th * layout.cell_w / layout.cell_h))
    mask = layout.template_mask(tw, th)

    stats = {p: os.stat(p) for p in paths}
    cached = {}
    cache_path = _cache_file(cache_dir, corpus_dir, layout, (tw, th)) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as z:
            for p, m, s, f in zip(z["paths"], z["mtimes"], z["sizes"], z["features"]):
                cached[str(p)] = (float(m), int(s), f)

    features = {}
    todo = []
    for p in paths:
        c = cached.get(p)
        st = stats[p]
        if c and c[0] == st.st_mtime and c[1] == st.st_size:
            features[p] = c[2]
        else:
            todo.append(p)

    if todo:
        log(f"  Computing colour features for {len(todo)} images "
            f"({len(paths) - len(todo)} cached) ...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for n, (p, f) in enumerate(zip(todo, ex.map(
                    lambda p: _feature(p, (tw, th), mask), todo)), 1):
                features[p] = np.full(3, np.nan) if f is None else f
                if n % 200 == 0:
                    log(f"    {n}/{len(todo)}")
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path,
                     paths=np.array(paths),
                     mtimes=np.array([stats[p].st_mtime for p in paths]),
                     sizes=np.array([stats[p].st_size for p in paths]),
                     features=np.array([features[p] for p in paths]))

    valid = [p for p in paths if not np.isnan(features[p]).any()]
    if len(valid) < len(paths):
        log(f"  Skipped {len(paths) - len(valid)} unreadable files")
    if max_pool and len(valid) > max_pool:
        rng = np.random.default_rng(seed)
        valid = sorted(rng.choice(valid, max_pool, replace=False).tolist())
        log(f"  (pool truncated to {max_pool} images)")
    return Pool(valid, np.array([features[p] for p in valid]).reshape(-1, 3))
