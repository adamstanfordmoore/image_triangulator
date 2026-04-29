# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`read_metadata.py` is a single-file CLI tool that extracts and pretty-prints metadata (EXIF, IPTC, XMP) from iPhone photos. It outputs JSON to stdout.

## Setup

```bash
brew install exiftool          # recommended — produces richest output
pip install -r requirements.txt
```

Python deps: `Pillow`, `exifread`, `piexif`, `pyheif`

## Running

```bash
python3 read_metadata.py /path/to/photo.jpg
python3 read_metadata.py /path/to/folder
```

Supported formats when scanning a directory: `.jpg`, `.jpeg`, `.heic`, `.png`, `.tiff`, `.dng`

## Architecture

Everything lives in `read_metadata.py`. The fallback chain in `process_file()` is:

1. **exiftool** (subprocess, `-j` flag → JSON) — used if `exiftool` is on PATH
2. **exifread** — pure-Python EXIF reader
3. **Pillow** — general image library; falls back to **pyheif** for `.heic` files if Pillow alone fails

`process_path()` dispatches between single-file and directory modes. Directory mode recurses one level (no nesting) and returns a dict keyed by file path.
