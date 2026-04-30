"""
calibration.py — intrinsic calibration data for known iPhone lenses.

Rear wide values measured via ChArUco calibration (34 images, RMS 0.86 px).
Front values measured via ChArUco calibration (18 images, RMS 0.36 px).
"""

import math
import numpy as np

# Keyed by the LensModel EXIF string (unique per physical camera assembly).
# Each entry: {'width': int, 'height': int, 'K': 3×3, 'dist': [k1,k2,p1,p2,k3]}
_KNOWN: dict[str, dict] = {
    "iPhone 16 back dual wide camera 5.96mm f/1.6": {
        "width":  5712,
        "height": 4284,
        # Measured via ChArUco calibration, 34 images, RMS 0.86 px
        "K":    np.array([[4397.47,    0.0, 2726.11],
                          [   0.0, 4392.91, 2131.66],
                          [   0.0,    0.0,    1.0]], dtype=np.float64),
        "dist": np.array([0.157882, -0.610850, 0.001754, -0.012938, 0.915894], dtype=np.float64),
    },
    "iPhone 16 front TrueDepth camera 2.69mm f/1.9": {
        "width":  4032,
        "height": 3024,
        # Measured via ChArUco calibration, 18 images, RMS 0.36 px
        "K":    np.array([[2725.71,    0.0, 2019.10],
                          [   0.0, 2725.03, 1517.44],
                          [   0.0,    0.0,    1.0]], dtype=np.float64),
        "dist": np.array([0.207011, -0.555202, 0.000661, 0.001165, 0.436016], dtype=np.float64),
    },
}


def get_calibration(
    lens_model: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Return (K, dist_coeffs) for the given lens at the requested resolution,
    or (None, None) if the lens is not in the known-calibrations table.

    K is scaled proportionally when the requested resolution differs from the
    calibrated resolution (e.g. 12MP vs 48MP capture mode).
    """
    entry = _KNOWN.get(lens_model)
    if entry is None:
        return None, None

    K = entry["K"].copy()
    cal_w = entry["width"]

    if width != cal_w:
        scale = width / cal_w
        K[0, 0] *= scale   # fx
        K[1, 1] *= scale   # fy
        K[0, 2] *= scale   # cx
        K[1, 2] *= scale   # cy

    return K, entry["dist"].copy()


def focal_from_fov(hfov_deg: float, width: int) -> float:
    """Compute focal length in pixels from EXIF horizontal FOV and image width."""
    return width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
