import math

import numpy as np
import pytest
from PIL import Image

from mosaic.color import srgb_to_lab
from mosaic.geometry import HexLayout, RectLayout, make_layout
from mosaic.match import match_cells, repetition_score
from mosaic.pool import build_pool
from mosaic.render import render, sample_targets
from mosaic import cli


@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_hex_ownership_is_nearest_center(orientation):
    lay = HexLayout(300, 200, 17.0, orientation)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 300, 5000), rng.uniform(0, 200, 5000)
    idx = lay.cell_index(x, y)
    assert (idx >= 0).all(), "every canvas point must be owned"
    d = np.hypot(lay.centers[None, :, 0] - x[:, None], lay.centers[None, :, 1] - y[:, None])
    assert (idx == d.argmin(axis=1)).all()
    # Owned points lie inside their hexagon.
    c = lay.centers[idx]
    assert (lay.edge_distance(x - c[:, 0], y - c[:, 1]) >= -1e-9).all()


@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_hex_template_area(orientation):
    lay = HexLayout(100, 100, 50.0, orientation)
    mask = lay.template_mask(400, 400)
    hex_area = 1.5 * math.sqrt(3) * 50 ** 2
    assert mask.mean() == pytest.approx(hex_area / (lay.cell_w * lay.cell_h), abs=0.01)


def test_hex_neighbors():
    lay = HexLayout(100, 100, 10.0)
    assert len(lay.neighbor_offsets(1)) == 6
    assert len(lay.neighbor_offsets(2)) == 18
    # Distance-1 neighbours have centres exactly one apothem*2 away.
    i = lay.index_of((2, 2))
    for dq, dr in lay.neighbor_offsets(1):
        j = lay.index_of((2 + dq, 2 + dr))
        assert np.hypot(*(lay.centers[i] - lay.centers[j])) == pytest.approx(2 * lay.inner)


def test_make_layout_cols():
    lay = make_layout("hex", 1000, 600, cols=50)
    assert lay.cell_w == pytest.approx(20.0)
    sq = make_layout("square", 1000, 600, cols=50, rows=20)
    assert (sq.cell_w, sq.cell_h) == (20.0, 30.0)
    assert len(sq) == 50 * 20


def test_lab_reference_values():
    np.testing.assert_allclose(srgb_to_lab([1, 1, 1]), [100, 0, 0], atol=0.01)
    np.testing.assert_allclose(srgb_to_lab([0, 0, 0]), [0, 0, 0], atol=0.01)
    np.testing.assert_allclose(srgb_to_lab([1, 0, 0]), [53.24, 80.09, 67.20], atol=0.05)


@pytest.mark.parametrize("matcher", ["greedy", "random"])
@pytest.mark.parametrize("shape", ["hex", "square"])
def test_match_respects_min_distance(shape, matcher):
    lay = make_layout(shape, 400, 300, cols=12)
    rng = np.random.default_rng(1)
    targets = np.zeros((len(lay), 3))  # every cell wants the same colour
    pool = rng.normal(0, 5, (200, 3))
    assigned, fallbacks = match_cells(lay, targets, pool, min_distance=3, candidates=200,
                                      matcher=matcher, tolerance=50)
    assert fallbacks == 0
    for i, (q, r) in enumerate(lay.coords.tolist()):
        for dq, dr in lay.neighbor_offsets(2):  # min_distance 3 => nothing within 2
            j = lay.index_of((q + dq, r + dr))
            if j is not None:
                assert assigned[i] != assigned[j]


def test_greedy_with_no_constraint_picks_best():
    lay = make_layout("hex", 200, 200, cols=5)
    targets = np.zeros((len(lay), 3))
    pool = np.array([[9.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0]])
    assigned, _ = match_cells(lay, targets, pool, min_distance=0, matcher="greedy")
    assert (assigned == 1).all()


@pytest.mark.parametrize("shape", ["hex", "square"])
def test_random_matcher_breaks_up_patterns(shape):
    """Uniform source + many similar pool images: greedy tiles a periodic
    lattice; random should not."""
    lay = make_layout(shape, 1200, 900, cols=40)
    rng = np.random.default_rng(2)
    pool = rng.normal(0, 2, (300, 3))       # 300 nearly identical images
    targets = np.zeros((len(lay), 3))
    greedy, _ = match_cells(lay, targets, pool, matcher="greedy", min_distance=4)
    rand, _ = match_cells(lay, targets, pool, matcher="random", min_distance=4, tolerance=10)
    assert repetition_score(lay, greedy) > 0.5
    assert repetition_score(lay, rand) < 0.1


def test_reuse_penalty_spreads_usage():
    lay = make_layout("hex", 600, 600, cols=20)
    targets = np.zeros((len(lay), 3))
    pool = np.array([[0.0, 0, 0]] + [[8.0 + i * 0.01, 0, 0] for i in range(50)])
    plain, _ = match_cells(lay, targets, pool, matcher="greedy", min_distance=0)
    spread, _ = match_cells(lay, targets, pool, matcher="greedy", min_distance=0,
                            reuse_penalty=1.0)
    assert np.bincount(plain).max() == len(lay)
    assert np.bincount(spread).max() < len(lay) / 5


@pytest.fixture
def solid_corpus(tmp_path):
    colors = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 210),
              "white": (250, 250, 250)}
    d = tmp_path / "corpus"
    d.mkdir()
    for name, c in colors.items():
        Image.new("RGB", (60, 40), c).save(d / f"{name}.jpg")
    (d / "bogus.jpg").write_bytes(b"not an image")
    return d


@pytest.mark.parametrize("shape", ["hex", "square"])
def test_end_to_end(tmp_path, solid_corpus, shape):
    # Source: left half red, right half blue.
    src = np.zeros((90, 160, 3), dtype=np.uint8)
    src[:, :80] = (220, 30, 30)
    src[:, 80:] = (30, 40, 210)
    Image.fromarray(src).save(tmp_path / "src.png")
    out = tmp_path / "out.png"
    rc = cli.main(["-s", str(tmp_path / "src.png"), "-c", str(solid_corpus), "-o", str(out),
                   "--shape", shape, "--cols", "8", "--scale", "2", "--min-distance", "0",
                   "--cache-dir", str(tmp_path / "cache")])
    assert rc == 0
    img = np.asarray(Image.open(out))
    assert img.shape == (180, 320, 3)  # SCALE honoured, rectangular output
    # Far left reads red, far right reads blue.
    assert img[:, :60, 0].mean() > 180 and img[:, :60, 2].mean() < 80
    assert img[:, -60:, 2].mean() > 180 and img[:, -60:, 0].mean() < 80
    # Second run is served from cache.
    assert any((tmp_path / "cache").iterdir())


def test_render_covers_canvas_without_gaps(solid_corpus):
    lay = HexLayout(120, 80, 9.0)
    pool = build_pool(str(solid_corpus), lay, cache_dir=None, log=lambda *_: None)
    white = pool.paths.index(str(solid_corpus / "white.jpg"))
    canvas = render(lay, np.full(len(lay), white), pool.paths, np.ones((len(lay), 3)),
                    grout_color="#ff00ff", log=lambda *_: None)
    # Magenta background must be fully covered.
    assert not ((canvas[..., 0] == 255) & (canvas[..., 1] == 0)).any()
