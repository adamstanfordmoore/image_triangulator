#!/usr/bin/env python3
"""
triangulate.py — main entry point for triangulator.

Estimate the real-world location of an object visible in two or more iPhone
photos using GPS position and compass heading from EXIF metadata.

Requires exiftool on PATH and: pip install Pillow matplotlib folium

Usage:
    python3 triangulate.py <image1> <image2> [image3 ...]

Each image opens in a window. Click the target object, then close the window.
Triangulation and a map are produced once all images are marked.
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from geo import extract_camera, get_exif, triangulate
from map_viz import show_map


# ── Interactive point selection ───────────────────────────────────────────────

_MAX_DISPLAY = (1400, 900)


def select_point(image_path: str, label: str):
    """
    Open image_path in a matplotlib window (EXIF orientation applied).
    User left-clicks to place a marker; right-click to clear; close to confirm.

    Returns (px, py, oriented_width, oriented_height) in the orientation-
    corrected coordinate space, or (None, None, None, None) if no point chosen.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    paths = [a for a in sys.argv[1:] if Path(a).is_file()]
    if len(paths) < 2:
        print("Usage: triangulate.py <image1> <image2> [image3 ...]")
        sys.exit(2)

    # Step 1 — user marks the target object in each image
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

    if len(clicks) < 2:
        print("\nNeed at least two marked images to triangulate.", file=sys.stderr)
        sys.exit(1)

    # Step 2 — build camera records from EXIF + click coordinates
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
