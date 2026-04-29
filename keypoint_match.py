"""
keypoint_match.py — wide-baseline keypoint matching for triangulator.

Uses SuperPoint (feature extractor) + LightGlue (matcher) to find a common
point across images taken from very different positions and angles.

Strategy: match all pairs, build correspondence tracks across images,
find the track that spans the most images, return its per-image centroid.

Requires: pip install git+https://github.com/cvg/LightGlue.git
          (torch is pulled in automatically)
"""

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from PIL import Image


_MAX_SIDE   = 1024   # resize long edge before feature extraction
_MAX_KP     = 4096   # max SuperPoint keypoints per image
_device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Image prep ────────────────────────────────────────────────────────────────

def _resize(pil_img):
    """Resize so the long edge ≤ _MAX_SIDE; return (resized_img, (sx, sy))."""
    w, h = pil_img.size
    scale = min(1.0, _MAX_SIDE / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img, (w / pil_img.size[0], h / pil_img.size[1])


def _to_tensor(pil_img):
    """PIL image → float32 grayscale tensor [1, H, W] in [0, 1] on device."""
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(gray).unsqueeze(0).to(_device)


# ── Track building (Union-Find) ───────────────────────────────────────────────

class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


# ── Public API ────────────────────────────────────────────────────────────────

def match_keypoints(pil_images):
    """
    Find the pixel location of the most prominent common structure across all
    images using SuperPoint + LightGlue.

    Matches every pair of images, builds cross-image correspondence tracks,
    then returns the per-image centroid of the track that spans the most images.

    pil_images : list of orientation-corrected PIL images.
    Returns a list the same length as pil_images:
      - (px, py) in original image coordinates if a match was found
      - None if that image had no keypoint in the winning track.
    """
    extractor = SuperPoint(max_num_keypoints=_MAX_KP).eval().to(_device)
    matcher   = LightGlue(features="superpoint").eval().to(_device)

    # ── 1. Extract features for all images ───────────────────────────────────
    print(f"  Extracting SuperPoint features ({_device}) …")
    scaled_imgs, scales = zip(*[_resize(img) for img in pil_images])
    tensors = [_to_tensor(img) for img in scaled_imgs]

    feats_all = []
    with torch.no_grad():
        for t in tensors:
            feats_all.append(extractor.extract(t.unsqueeze(0)))

    # kps_all[i] : (N_i, 2) numpy array of keypoint coords in SCALED image space
    kps_all = [f["keypoints"][0].cpu().numpy() for f in feats_all]

    # ── 2. Match every pair, build correspondence tracks ──────────────────────
    print(f"  Matching {len(pil_images)} images ({len(pil_images)*(len(pil_images)-1)//2} pairs) …")
    uf = _UnionFind()

    with torch.no_grad():
        for i in range(len(pil_images)):
            for j in range(i + 1, len(pil_images)):
                result = matcher({"image0": feats_all[i], "image1": feats_all[j]})
                result = rbd(result)
                matches = result["matches"].cpu().numpy()   # (M, 2)
                if len(matches) == 0:
                    continue
                for ki, kj in matches:
                    uf.union((i, int(ki)), (j, int(kj)))

    # ── 3. Group tracks by root; find the one spanning the most images ────────
    root_to_nodes = defaultdict(list)
    for i in range(len(pil_images)):
        for ki in range(len(kps_all[i])):
            root_to_nodes[uf.find((i, ki))].append((i, ki))

    def _image_span(nodes):
        return len({img_idx for img_idx, _ in nodes})

    best_track = max(root_to_nodes.values(), key=_image_span)
    span = _image_span(best_track)
    print(f"  Best track spans {span}/{len(pil_images)} images "
          f"({len(best_track)} keypoints).")

    if span < 2:
        print("  Warning: no consistent match found across images.")
        return [None] * len(pil_images)

    # ── 4. Per image: centroid of all keypoints in the winning track ──────────
    # Group track keypoints by image
    track_by_image = defaultdict(list)
    for img_idx, ki in best_track:
        track_by_image[img_idx].append(kps_all[img_idx][ki])

    result_coords = []
    for i in range(len(pil_images)):
        if i not in track_by_image:
            result_coords.append(None)
        else:
            pts = np.array(track_by_image[i])          # (K, 2) in scaled space
            cx, cy = pts.mean(axis=0)
            sx, sy = scales[i]
            result_coords.append((int(cx * sx), int(cy * sy)))

    return result_coords


# ── Visualisation ─────────────────────────────────────────────────────────────

def show_match_grid(pil_images, pixel_coords, labels=None):
    """
    Display all images in a matplotlib grid with the matched point marked.
    Blocks until the user closes the window.
    """
    _MAX_DISPLAY = (900, 700)
    n    = len(pil_images)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.2))
    fig.canvas.manager.set_window_title("LightGlue matches — close to confirm")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (img, coord) in enumerate(zip(pil_images, pixel_coords)):
        ax = axes_flat[i]
        disp = img.copy()
        disp.thumbnail(_MAX_DISPLAY, Image.LANCZOS)
        sx = disp.size[0] / img.size[0]
        sy = disp.size[1] / img.size[1]

        ax.imshow(disp, origin="upper")
        ax.set_axis_off()

        name = labels[i] if labels else f"Image {i}"
        if coord is not None:
            dx, dy = coord[0] * sx, coord[1] * sy
            ax.plot(dx, dy, "r+", markersize=22, markeredgewidth=2.5, zorder=5)
            ax.plot(dx, dy, "ro", markersize=9, markerfacecolor="none",
                    markeredgewidth=1.5, zorder=5)
            ax.set_title(f"{name}\n({coord[0]}, {coord[1]})", fontsize=9)
        else:
            ax.set_title(f"{name}\nno match", fontsize=9, color="#CC0000")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "LightGlue matched keypoints  —  close window to proceed",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    plt.show()
