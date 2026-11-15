"""sRGB -> CIELAB conversion (D65), so colour distances track perception."""

import numpy as np

_RGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_EPS = 216 / 24389
_KAPPA = 24389 / 27


def srgb_to_lab(rgb) -> np.ndarray:
    """Convert sRGB floats in [0, 1], shape (..., 3), to CIELAB."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _RGB_TO_XYZ.T) / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)
