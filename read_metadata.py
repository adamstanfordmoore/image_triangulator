#!/usr/bin/env python3
"""
read_metadata.py

Read and pretty-print metadata from image files (iPhone photos) including EXIF, IPTC, XMP.
Tries exiftool if available for the most complete output. Falls back to pure-Python libraries.

Usage:
    python read_metadata.py /path/to/image.jpg
    python read_metadata.py /path/to/folder

Outputs JSON to stdout.
"""

import sys
import os
import json
import subprocess
from pathlib import Path


def which(cmd):
    from shutil import which as _which
    return _which(cmd)


def run_exiftool(path):
    # Run exiftool -j for JSON output
    try:
        res = subprocess.run(["exiftool", "-j", str(path)], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        # exiftool -j returns a list even for single files
        if isinstance(data, list) and len(data) == 1:
            return data[0]
        return data
    except Exception as e:
        return {"error": f"exiftool failed: {e}"}


def read_with_exifread(path):
    try:
        import exifread
    except Exception as e:
        return {"error": f"exifread not installed: {e}"}
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        return {k: str(v) for k, v in tags.items()}
    except Exception as e:
        return {"error": str(e)}


def read_with_pillow(path):
    try:
        from PIL import Image
        img = Image.open(path)
        info = {}
        try:
            exif = img._getexif()
            if exif:
                info["exif"] = exif
        except Exception:
            pass
        # Basic info
        info["format"] = getattr(img, "format", None)
        info["mode"] = getattr(img, "mode", None)
        info["size"] = getattr(img, "size", None)
        return info
    except Exception as e:
        # Try pyheif for HEIC images
        try:
            import pyheif
            from PIL import Image
            heif_file = pyheif.read(str(path))
            img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
            )
            info = {"format": "HEIC", "size": getattr(img, "size", None), "mode": getattr(img, "mode", None)}
            return info
        except Exception:
            return {"error": str(e)}


def process_path(p):
    p = Path(p)
    if p.is_dir():
        out = {}
        for child in sorted(p.iterdir()):
            if child.is_file() and child.suffix.lower() in (".jpg", ".jpeg", ".heic", ".png", ".tiff", ".dng"):
                out[str(child)] = process_file(child)
        return out
    elif p.is_file():
        return process_file(p)
    else:
        return {"error": "path not found"}


def process_file(p):
    p = Path(p)
    # Prefer exiftool if available
    if which("exiftool"):
        res = run_exiftool(p)
        return res
    # Try exifread
    res = read_with_exifread(p)
    if "error" not in res:
        return res
    # Try pillow
    res = read_with_pillow(p)
    return res


def _json_default(obj):
    # IFDRational, bytes, and other non-serializable types from Pillow/exifread
    try:
        return float(obj)
    except Exception:
        return str(obj)


def main():
    if len(sys.argv) < 2:
        print("Usage: read_metadata.py <path-to-image-or-directory>")
        sys.exit(2)
    path = sys.argv[1]
    res = process_path(path)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
