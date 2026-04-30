#!/usr/bin/env python3
"""
calibrate.py — ChArUco-based camera intrinsic calibration.

Generate a board to display on screen:
    python3 calibrate.py --generate              # saves charuco_board.png
    python3 calibrate.py --generate --out board.png --squares-x 7 --squares-y 5

Calibrate from a folder of iPhone photos of that board:
    python3 calibrate.py /path/to/photos/

Capture ~15-20 photos at varied angles and distances with the same camera
you use for triangulation. Transfer to Mac (AirDrop, iCloud, cable) first.
"""

import argparse
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


# ── Board parameters ──────────────────────────────────────────────────────────

def make_board(squares_x: int, squares_y: int) -> tuple:
    """Return (board, dictionary) for a ChArUco board."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=1.0,   # arbitrary units — only intrinsics matter
        markerLength=0.75,
        dictionary=dictionary,
    )
    return board, dictionary


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(out_path: Path, squares_x: int, squares_y: int) -> None:
    board, _ = make_board(squares_x, squares_y)

    # Size: 150 px per square, white border = 1 square on each side
    px_per_square = 150
    w = squares_x * px_per_square
    h = squares_y * px_per_square
    img = board.generateImage((w, h), marginSize=px_per_square // 2)

    cv2.imwrite(str(out_path), img)
    print(f"Board saved → {out_path}")
    print(f"Open full-screen (Preview → Shift+Cmd+F) before photographing.")
    print(f"Board: {squares_x}×{squares_y} squares  ({squares_x-1}×{squares_y-1} inner corners)")


# ── Calibrate ─────────────────────────────────────────────────────────────────

def calibrate(image_dir: Path, squares_x: int, squares_y: int) -> None:
    board, _ = make_board(squares_x, squares_y)
    detector = aruco.CharucoDetector(board)

    all_obj_pts = []
    all_img_pts = []
    img_size = None

    exts = {".jpg", ".jpeg", ".png", ".heic"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        sys.exit(f"No images found in {image_dir}")

    found = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip (unreadable): {path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < 4:
            print(f"  skip (too few corners): {path.name}")
            continue

        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        all_obj_pts.append(obj_pts)
        all_img_pts.append(img_pts)
        found += 1
        print(f"  found {len(charuco_ids):2d} corners: {path.name}")

    if found < 5:
        sys.exit(f"Need at least 5 usable images; only got {found}.")

    print(f"\nCalibrating with {found} images …")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        all_obj_pts, all_img_pts, img_size, None, None
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist[0, :4]
    k3 = dist[0, 4] if dist.shape[1] > 4 else 0.0
    w, h = img_size

    print(f"\nRMS reprojection error: {rms:.4f} px  (< 1.0 is good)")
    print(f"\n# ── Paste into calibration.py ────────────────────────────────")
    print(f'    "YOUR_LENS_MODEL_FROM_EXIF": {{')
    print(f'        "width":  {w},')
    print(f'        "height": {h},')
    print(f'        "K":    np.array([[{fx:.2f},    0.0, {cx:.2f}],')
    print(f'                          [   0.0, {fy:.2f}, {cy:.2f}],')
    print(f'                          [   0.0,    0.0,    1.0]], dtype=np.float64),')
    print(f'        "dist": np.array([{k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f}], dtype=np.float64),')
    print(f'    }},')
    print(f'\n# Get your LensModel string with:')
    print(f'#   exiftool -LensModel /path/to/photo.jpg')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="ChArUco camera calibration — generate board or calibrate from photos."
    )
    ap.add_argument("image_dir", nargs="?", type=Path,
                    help="Directory of iPhone photos (omit when using --generate)")
    ap.add_argument("--generate", action="store_true",
                    help="Generate the ChArUco board PNG instead of calibrating")
    ap.add_argument("--out", type=Path, default=Path("charuco_board.png"),
                    help="Output path for --generate (default: charuco_board.png)")
    ap.add_argument("--squares-x", type=int, default=7,
                    help="Squares across (default 7)")
    ap.add_argument("--squares-y", type=int, default=5,
                    help="Squares tall (default 5)")
    args = ap.parse_args()

    if args.generate:
        generate(args.out, args.squares_x, args.squares_y)
    else:
        if args.image_dir is None:
            ap.error("Provide a photo directory, or use --generate to create the board.")
        if not args.image_dir.is_dir():
            sys.exit(f"Not a directory: {args.image_dir}")
        calibrate(args.image_dir, args.squares_x, args.squares_y)


if __name__ == "__main__":
    main()
