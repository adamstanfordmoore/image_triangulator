"""
geo.py — geodetic math for triangulator.

Pure functions: no I/O, no GUI, no side effects (subprocess for exiftool only).
"""

import json
import math
import re
import subprocess
from pathlib import Path


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

    width = display_width if display_width is not None else int(meta["ImageWidth"])
    height = display_height if display_height is not None else int(meta["ImageHeight"])

    # GPSImgDirection = direction the back of the phone faces.
    # For a back camera that equals the lens/scene direction directly.
    # For a front (selfie) camera the back still faces the background, so
    # GPSImgDirection is also correct as-is — no correction needed.
    # (Adding 180° would point toward the photographer's face, not the scene.)

    # Pixel offset from image centre → horizontal bearing adjustment
    if px is not None and py is not None:
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


# ── Ray intersection ──────────────────────────────────────────────────────────

def bearing_to_unit(bearing_deg):
    """Bearing (°CW from True North)  →  (dx, dy) unit vector in E/N space."""
    r = math.radians(bearing_deg)
    return math.sin(r), math.cos(r)


def intersect_rays(p1, b1, p2, b2):
    """
    Intersect two 2-D rays in local flat-Earth coordinates.

    p1, p2 : (x, y) origins in metres
    b1, b2 : bearings in degrees

    Returns (x, y, t1, t2) where t1/t2 are signed distances along each ray.
    Negative t means the intersection is behind that camera.
    Returns None if rays are parallel.
    """
    d1x, d1y = bearing_to_unit(b1)
    d2x, d2y = bearing_to_unit(b2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]

    det = d2x * d1y - d1x * d2y
    if abs(det) < 1e-10:
        return None

    t1 = (d2x * dy - d2y * dx) / det
    t2 = (d1x * dy - dx * d1y) / det
    return p1[0] + t1 * d1x, p1[1] + t1 * d1y, t1, t2


def triangulate(cameras):
    """
    Given a list of camera records (from extract_camera), return the estimated
    target position as (lat, lon).

    For 2 cameras: exact intersection.
    For 3+: average of all pairwise intersections (least-squares approximation).

    Also returns a list of warning strings for any degenerate ray pairs.
    """
    origin_lat = sum(c["lat"] for c in cameras) / len(cameras)
    origin_lon = sum(c["lon"] for c in cameras) / len(cameras)
    points = [to_xy(c["lat"], c["lon"], origin_lat, origin_lon) for c in cameras]

    hits = []
    warnings = []

    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            result = intersect_rays(
                points[i], cameras[i]["bearing"],
                points[j], cameras[j]["bearing"],
            )
            if result is None:
                warnings.append(
                    f"Rays from {cameras[i]['file']} and {cameras[j]['file']} "
                    f"are parallel — skipping pair."
                )
                continue
            ix, iy, t1, t2 = result
            if t1 < 0:
                warnings.append(
                    f"Intersection is behind {cameras[i]['file']} "
                    f"({t1:.0f} m) — check bearing/pixel."
                )
            if t2 < 0:
                warnings.append(
                    f"Intersection is behind {cameras[j]['file']} "
                    f"({t2:.0f} m) — check bearing/pixel."
                )
            hits.append((ix, iy))

    if not hits:
        return None, None, warnings

    est_x = sum(h[0] for h in hits) / len(hits)
    est_y = sum(h[1] for h in hits) / len(hits)
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
