# triangulator

Estimate the real-world location of an object by clicking on it in two or more iPhone photos. Uses GPS position, compass heading, and field of view from EXIF metadata to cast bearing rays from each camera and find their intersection.

Tested at ~300 m accuracy on the Golden Gate Bridge using 6 photos from different locations around the bay. Accuracy improves with more images and a wider baseline between shooting positions.

## How it works

Each photo contributes a ray: starting from the camera's GPS position, aimed along the bearing derived from the compass heading adjusted for where the object sits in the frame. Two rays intersect at a point estimate; three or more rays are averaged across all pairwise intersections.

Front/selfie cameras are handled automatically — `GPSImgDirection` always records the direction the back of the phone faces, which is the background/scene direction regardless of which lens captured the image.

## Setup

```bash
brew install exiftool
pip install -r requirements.txt
```

## Usage

```bash
python3 triangulate.py photo1.jpg photo2.jpg [photo3.jpg ...]
```

Each image opens in a window. **Left-click** to mark the target object, **right-click** to clear, then **close the window** to confirm. After all images are marked, the estimated position is printed and an interactive map opens in the browser showing camera positions, bearing rays with uncertainty cones, and the result.

## Map

The folium map includes an OpenStreetMap / Esri satellite layer toggle (top-right corner). Each camera is shown as a coloured dot with its bearing ray (dashed) and a ±5° uncertainty cone. The estimated target position is shown in gold.

## Files

| File | Purpose |
|------|---------|
| `triangulate.py` | CLI entry point and matplotlib click UI |
| `geo.py` | Geodetic math: EXIF parsing, ray intersection, coordinate projection |
| `map_viz.py` | Folium map rendering |
| `read_metadata.py` | Standalone utility to dump all EXIF from an image or folder |

## Accuracy

Dominant error sources, largest first:

1. **Compass heading** — iPhone compass has ~5–10° uncertainty; at 5 km range, 1° ≈ 90 m of position error
2. **Click precision** — hard to click the exact same pixel on a distant object across multiple photos
3. **GPS position** — typically ±3–7 m, minor compared to heading error

Adding more images with diverse shooting angles reduces the effect of individual bearing errors.
