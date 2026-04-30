#!/usr/bin/env python3
"""
bearing_delta_heatmap.py — visualise per-pixel bearing error introduced by
skipping camera calibration (linear FOV model vs calibrated undistortion).

Delta = calibrated_bearing_offset − uncalibrated_bearing_offset  (degrees)
"""

import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from calibration import get_calibration

CAMERAS = [
    dict(
        label="Rear wide\n(5.96mm f/1.6)",
        lens="iPhone 16 back dual wide camera 5.96mm f/1.6",
        width=5712, height=4284, hfov=69.4,
    ),
    dict(
        label="Front TrueDepth\n(2.69mm f/1.9)",
        lens="iPhone 16 front TrueDepth camera 2.69mm f/1.9",
        width=4032, height=3024, hfov=76.1,
    ),
]

STEP = 16   # compute every Nth pixel — 16 is fast and plenty of resolution


def compute_delta(lens, width, height, hfov, step=STEP):
    K, dist = get_calibration(lens, width, height)

    px = np.arange(0, width,  step, dtype=np.float32)
    py = np.arange(0, height, step, dtype=np.float32)
    gx, gy = np.meshgrid(px, py)

    # Uncalibrated: linear FOV model (current code without calibration)
    dx = gx - width / 2.0
    b_uncal = (dx / width) * hfov          # degrees, signed

    # Calibrated: undistort → normalised coord → atan
    pts = np.stack([gx, gy], axis=-1).reshape(-1, 1, 2)
    xn = cv2.undistortPoints(pts, K, dist)[:, 0, 0].reshape(gx.shape)
    b_cal = np.degrees(np.arctan(xn))      # degrees, signed

    return b_cal - b_uncal, gx, gy


fig = plt.figure(figsize=(15, 5))
fig.suptitle("Bearing error from skipping calibration  (calibrated − linear FOV model)",
             fontsize=13)

# Two image axes + one narrow colorbar axis
ax1 = fig.add_axes([0.05, 0.1, 0.40, 0.78])
ax2 = fig.add_axes([0.50, 0.1, 0.40, 0.78])
cax = fig.add_axes([0.93, 0.1, 0.02, 0.78])
axes = [ax1, ax2]

vmax = 0.0
deltas = []
for cam in CAMERAS:
    d, gx, gy = compute_delta(cam["lens"], cam["width"], cam["height"], cam["hfov"])
    deltas.append((d, gx, gy))
    vmax = max(vmax, np.abs(d).max())

for ax, cam, (delta, gx, gy) in zip(axes, CAMERAS, deltas):
    im = ax.imshow(
        delta,
        extent=[0, cam["width"], cam["height"], 0],
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
    )
    ax.set_title(cam["label"], fontsize=11)
    ax.set_xlabel("Pixel x")
    ax.set_ylabel("Pixel y")

    # Annotate corners and centre with their delta values
    for (px, py, lbl) in [
        (0,               0,                "TL"),
        (cam["width"]-1,  0,                "TR"),
        (0,               cam["height"]-1,  "BL"),
        (cam["width"]-1,  cam["height"]-1,  "BR"),
        (cam["width"]//2, cam["height"]//2, "C"),
    ]:
        xi = min(int(px // STEP), delta.shape[1] - 1)
        yi = min(int(py // STEP), delta.shape[0] - 1)
        val = delta[yi, xi]
        ax.text(px, py, f"{lbl}\n{val:+.2f}°",
                ha="center", va="center", fontsize=7,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Draw the zero-delta contour
    ax.contour(gx, gy, delta, levels=[0], colors="white", linewidths=1.2, linestyles="--")

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Δ bearing (°)  [+ve = cal pushes ray rightward]", fontsize=10)

plt.show()
