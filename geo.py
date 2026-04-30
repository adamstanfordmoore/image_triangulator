"""
geo.py — geodetic math for triangulator.

Pure functions: no I/O, no GUI, no side effects (subprocess for exiftool only).
"""

import json
import math
import re
import subprocess

import cv2
import numpy as np
from pathlib import Path

from calibration import get_calibration


# ── EXIF / metadata ───────────────────────────────────────────────────────────

def get_exif(path: str) -> dict:
    result = subprocess.run(
        ["exiftool", "-j", path],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return data[0] if isinstance(data, list) else data


def parse_gps_dms(s: str) -> float:
    """'37 deg 49' 40.29" N'  →  +37.827858"""
    m = re.match(r"([\d.]+) deg ([\d.]+)' ([\d.]+)\" ([NSEW])", s)
    if not m:
        raise ValueError(f"Cannot parse GPS string: {s!r}")
    deg, mn, sec, ref = float(m[1]), float(m[2]), float(m[3]), m[4]
    dd = deg + mn / 60 + sec / 3600
    return -dd if ref in ("S", "W") else dd


def parse_fov(s) -> float:
    return float(str(s).split()[0])


def extract_camera(meta: dict, px=None, py=None,
                   display_width=None, display_height=None) -> dict:
    """
    Build a camera record from exiftool metadata.

    display_width / display_height should be the dimensions of the
    orientation-corrected image (after ImageOps.exif_transpose) so that the
    horizontal pixel offset has the correct sign for all rotation cases.
    """
    lat = parse_gps_dms(meta["GPSLatitude"])
    lon = parse_gps_dms(meta["GPSLongitude"])
    bearing = float(meta["GPSImgDirection"])
    hfov = parse_fov(meta["FOV"])
    is_front = "Front" in meta.get("CameraType", "")
    lens_model = meta.get("LensModel", "")

    width = display_width if display_width is not None else int(meta["ImageWidth"])
    height = display_height if display_height is not None else int(meta["ImageHeight"])

    # GPSImgDirection = direction the back of the phone faces.
    # For a back camera that equals the lens/scene direction directly.
    # For a front (selfie) camera the back still faces the background, so
    # GPSImgDirection is also correct as-is — no correction needed.
    # (Adding 180° would point toward the photographer's face, not the scene.)

    # Pixel offset from image centre → horizontal bearing adjustment
    calibrated = False
    if px is not None and py is not None:
        K, dist = get_calibration(lens_model, width, height)
        if K is not None:
            # Undistort the clicked pixel; cv2.undistortPoints returns
            # normalized image coordinates (x/fx, y/fy) with distortion removed.
            pt = cv2.undistortPoints(
                np.array([[[float(px), float(py)]]], dtype=np.float32), K, dist
            )
            xn = float(pt[0, 0, 0])
            bearing = (bearing + math.degrees(math.atan(xn))) % 360.0
            calibrated = True
        else:
            dx_pixels = float(px) - width / 2.0
            bearing = (bearing + (dx_pixels / width) * hfov) % 360.0

    return {
        "file": meta.get("FileName", Path(meta.get("SourceFile", "?")).name),
        "lat": lat,
        "lon": lon,
        "bearing": bearing,
        "hfov": hfov,
        "width": width,
        "height": height,
        "is_front": is_front,
        "lens_model": lens_model,
        "calibrated": calibrated,
        "px": px,
        "py": py,
    }


# ── Flat-Earth projection (accurate to ~0.1 % over 10 km) ────────────────────

_M_PER_DEG_LAT = 111_320.0


def to_xy(lat, lon, origin_lat, origin_lon):
    """lat/lon  →  local (x=East, y=North) in metres."""
    scale = _M_PER_DEG_LAT * math.cos(math.radians(origin_lat))
    return (lon - origin_lon) * scale, (lat - origin_lat) * _M_PER_DEG_LAT


def from_xy(x, y, origin_lat, origin_lon):
    """Local metres  →  lat/lon."""
    scale = _M_PER_DEG_LAT * math.cos(math.radians(origin_lat))
    return origin_lat + y / _M_PER_DEG_LAT, origin_lon + x / scale


# ── Ray / least-squares intersection ─────────────────────────────────────────

def bearing_to_unit(bearing_deg):
    """Bearing (°CW from True North)  →  (dx, dy) unit vector in E/N space."""
    r = math.radians(bearing_deg)
    return math.sin(r), math.cos(r)


def _least_squares_point(points, bearings):
    """
    Find the point P* minimising the sum of squared perpendicular distances
    to all bearing rays (closed-form linear least squares).

    For each ray with origin p_i and unit direction d_i, the perpendicular
    projector is A_i = I - d_i dᵢᵀ. The optimal point satisfies:

        (Σ A_i) P* = Σ A_i p_i

    Returns (x, y) in local metres.  Raises np.linalg.LinAlgError if the
    system is singular (all rays parallel).
    """
    A_sum  = np.zeros((2, 2))
    Ap_sum = np.zeros(2)

    for (px, py), bearing in zip(points, bearings):
        d = np.array(bearing_to_unit(bearing))
        p = np.array([px, py])
        A = np.eye(2) - np.outer(d, d)   # perpendicular projector
        A_sum  += A
        Ap_sum += A @ p

    # Use lstsq for robustness when rays are nearly parallel
    P, _, rank, _ = np.linalg.lstsq(A_sum, Ap_sum, rcond=None)
    if rank < 2:
        raise np.linalg.LinAlgError("All rays are parallel — cannot triangulate.")
    return float(P[0]), float(P[1])


def triangulate(cameras):
    """
    Given a list of camera records (from extract_camera), return the estimated
    target position as (lat, lon) using closed-form least-squares minimisation
    of perpendicular distances to all bearing rays.

    Also returns a list of warning strings for cameras whose estimated point
    falls behind them (negative along-ray distance).
    """
    origin_lat = sum(c["lat"] for c in cameras) / len(cameras)
    origin_lon = sum(c["lon"] for c in cameras) / len(cameras)
    points   = [to_xy(c["lat"], c["lon"], origin_lat, origin_lon) for c in cameras]
    bearings = [c["bearing"] for c in cameras]

    warnings = []
    try:
        est_x, est_y = _least_squares_point(points, bearings)
    except np.linalg.LinAlgError as e:
        warnings.append(str(e))
        return None, None, warnings

    # Warn if the estimate is behind any camera (along-ray distance < 0)
    for cam, (px, py) in zip(cameras, points):
        d = np.array(bearing_to_unit(cam["bearing"]))
        t = float(np.dot(np.array([est_x - px, est_y - py]), d))
        if t < 0:
            warnings.append(
                f"Estimated point is behind {cam['file']} "
                f"({t:.0f} m) — check bearing/pixel."
            )

    est_lat, est_lon = from_xy(est_x, est_y, origin_lat, origin_lon)
    return est_lat, est_lon, warnings


# ── Spherical geometry helpers (used by map visualisation) ───────────────────

def point_at_bearing(lat, lon, bearing_deg, distance_m):
    """Destination lat/lon given origin, bearing (°CW from N), and distance (m)."""
    R = 6_371_000.0
    b = math.radians(bearing_deg)
    d = distance_m / R
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                     math.cos(lat1) * math.sin(d) * math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1),
                              math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def cone_polygon(lat, lon, bearing, half_angle_deg, distance_m, steps=40):
    """
    Return a [lat, lon] polygon forming a bearing uncertainty fan/cone.
    Apex at (lat, lon), spanning ±half_angle_deg around bearing out to distance_m.
    """
    pts = [[lat, lon]]
    for i in range(steps + 1):
        b = bearing - half_angle_deg + 2 * half_angle_deg * i / steps
        p = point_at_bearing(lat, lon, b, distance_m)
        pts.append([p[0], p[1]])
    pts.append([lat, lon])
    return pts
