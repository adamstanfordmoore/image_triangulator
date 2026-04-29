read_metadata.py

Reads metadata from image files (iPhone photos). Usage:

1. Install dependencies (optional, exiftool is recommended):

```bash
# Using Homebrew for exiftool (recommended):
brew install exiftool

# Python deps:
python3 -m pip install -r requirements.txt
```

2. Run:

```bash
python3 read_metadata.py /path/to/photo.jpg
python3 read_metadata.py /path/to/folder
```

Notes:

- If `exiftool` is installed it will be used and produces the richest output.
- The script falls back to Python libraries when `exiftool` isn't available.
