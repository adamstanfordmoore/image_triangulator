#!/usr/bin/env python3
"""
triangulate.py — main entry point for triangulator.

Estimate the real-world location of an object visible in two or more iPhone
photos using GPS position and compass heading from EXIF metadata.

Requires exiftool on PATH and: pip install Pillow matplotlib folium opencv-python

Usage:
    # Manual — click the target in each image window
    python3 triangulate.py <image1> <image2> [image3 ...]

    # Auto — SIFT keypoint matching finds the common point automatically
    python3 triangulate.py --auto <image1> <image2> [image3 ...]

Close each image window to confirm the selection and move to the next.
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from geo import extract_camera, get_exif, triangulate
from keypoint_match import match_keypoints, show_match_grid
from map_viz import show_map


# ── Image loading ─────────────────────────────────────────────────────────────

def load_oriented(path: str):
    """Open an image and apply its EXIF orientation."""
    img = Image.open(path)
    return ImageOps.exif_transpose(img)


# ── Manual point selection ────────────────────────────────────────────────────

_MAX_DISPLAY = (1400, 900)


def select_point(image_path: str, label: str):
    """
    Open image_path in a matplotlib window.
    Left-click to mark the target; right-click to clear; close to confirm.

    Returns (px, py, oriented_w, oriented_h) or (None, None, None, None).
    """
    img = load_oriented(image_path)
    oriented_w, oriented_h = img.size

    img.thumbnail(_MAX_DISPLAY, Image.LANCZOS)
    disp_w, disp_h = img.size
    scale_x = oriented_w / disp_w
    scale_y = oriented_h / disp_h

    state = {"point": None, "marker": None}

    fig, ax = plt.subplots(figsize=(disp_w / 96, disp_h / 96))
    fig.canvas.manager.set_window_title(label)
    ax.imshow(img, origin="upper")
    ax.set_axis_off()
    ax.set_title(
        f"{label}  —  click the target object  (right-click to clear)",
        fontsize=11, pad=8,
    )
    fig.tight_layout(pad=0.5)

    def _place_marker(dx, dy):
        if state["marker"]:
            for a in state["marker"]:
                a.remove()
        h = ax.plot(dx, dy, "r+", markersize=22, markeredgewidth=2.5, zorder=5)
        c = ax.plot(dx, dy, "ro", markersize=8, markerfacecolor="none",
                    markeredgewidth=1.5, zorder=5)
        state["marker"] = h + c
        ax.set_title(
            f"{label}  —  ({int(dx * scale_x)}, {int(dy * scale_y)})  •  close to confirm",
            fontsize=11, pad=8,
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            state["point"] = (event.xdata, event.ydata)
            _place_marker(event.xdata, event.ydata)
        elif event.button == 3:
            state["point"] = None
            if state["marker"]:
                for a in state["marker"]:
                    a.remove()
                state["marker"] = None
            ax.set_title(
                f"{label}  —  click the target object  (right-click to clear)",
                fontsize=11, pad=8,
            )
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if state["point"] is None:
        return None, None, None, None
    dx, dy = state["point"]
    return int(dx * scale_x), int(dy * scale_y), oriented_w, oriented_h


# ── Point collection strategies ───────────────────────────────────────────────

def collect_manual(paths):
    """Open each image and ask the user to click the target. Returns clicks dict."""
    clicks = {}
    for path in paths:
        name = Path(path).name
        print(f"\nOpening {name} — click the target object, then close the window.")
        px, py, ow, oh = select_point(path, name)
        if px is None:
            print(f"  No point selected — skipping {name}.")
        else:
            print(f"  Selected pixel ({px}, {py})")
            clicks[path] = (px, py, ow, oh)
    return clicks


def collect_auto(paths):
    """
    Use SIFT keypoint matching to automatically find a common point across all
    images, show the result grid, then return a clicks dict.

    For any image where no match was found, falls back to manual clicking.
    """
    print("\nLoading images and detecting keypoints…")
    images = [load_oriented(p) for p in paths]
    labels = [Path(p).name for p in paths]

    coords = match_keypoints(images)

    # Show the matched keypoints for user confirmation
    print("Displaying matched keypoints — close the window to continue.")
    show_match_grid(images, coords, labels=labels)

    clicks = {}
    for path, img, coord, label in zip(paths, images, coords, labels):
        ow, oh = img.size
        if coord is not None:
            px, py = coord
            print(f"  {label}: auto ({px}, {py})")
            clicks[path] = (px, py, ow, oh)
        else:
            # Fall back to manual for images with no auto match
            print(f"\n  No auto match for {label} — opening for manual selection.")
            px, py, ow, oh = select_point(path, label)
            if px is not None:
                print(f"  {label}: manual ({px}, {py})")
                clicks[path] = (px, py, ow, oh)
            else:
                print(f"  Skipping {label}.")

    return clicks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    auto_mode = "--auto" in args
    paths = [a for a in args if a != "--auto" and Path(a).is_file()]

    if len(paths) < 2:
        print("Usage: triangulate.py [--auto] <image1> <image2> [image3 ...]")
        sys.exit(2)

    # Step 1 — collect pixel coordinates
    clicks = collect_auto(paths) if auto_mode else collect_manual(paths)

    if len(clicks) < 2:
        print("\nNeed at least two marked images to triangulate.", file=sys.stderr)
        sys.exit(1)

    # Step 2 — build camera records from EXIF + pixel coordinates
    print()
    cameras = []
    for path, (px, py, ow, oh) in clicks.items():
        meta = get_exif(path)
        cam = extract_camera(meta, px, py, display_width=ow, display_height=oh)
        cameras.append(cam)
        print(f"  {cam['file']}")
        print(f"    GPS     : {cam['lat']:.6f}°N, {abs(cam['lon']):.6f}°W")
        print(f"    Bearing : {cam['bearing']:.2f}° True North"
              + ("  (front/selfie camera)" if cam["is_front"] else ""))
        print(f"    FOV     : {cam['hfov']:.1f}°   pixel ({px}, {py}) "
              f"in {cam['width']}×{cam['height']}")

    # Step 3 — triangulate
    est_lat, est_lon, warnings = triangulate(cameras)

    for w in warnings:
        print(f"\nWarning: {w}")

    if est_lat is None:
        print("\nNo valid intersection found.", file=sys.stderr)
        sys.exit(1)

    # Step 4 — report
    print()
    print("── Result " + "─" * 50)
    print(f"  Estimated position : {est_lat:.6f}°N, {abs(est_lon):.6f}°W")
    print(f"  Google Maps        : https://maps.google.com/?q={est_lat:.6f},{est_lon:.6f}")
    print()
    for cam in cameras:
        dist = math.hypot(
            (est_lat - cam["lat"]) * 111_320,
            (est_lon - cam["lon"]) * 111_320 * math.cos(math.radians(cam["lat"])),
        )
        print(f"  Distance from {cam['file']}: {dist:.0f} m")

    # Step 5 — map
    print()
    show_map(cameras, est_lat, est_lon)


if __name__ == "__main__":
    main()
