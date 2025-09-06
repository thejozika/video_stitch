import os
import io
import mimetypes
import shutil
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so we can import main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import app


FILES_DIR = Path(__file__).resolve().parents[1] / "files"
OUTPUT_VIDEO = FILES_DIR / "stitched_from_test.mp4"


def _is_audio(path: Path) -> bool:
    return path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def test_stitch_creates_video_from_files_folder():
    assert FILES_DIR.exists(), f"Expected directory not found: {FILES_DIR}"

    audio_paths = sorted([p for p in FILES_DIR.iterdir() if p.is_file() and _is_audio(p)])
    image_paths = sorted([p for p in FILES_DIR.iterdir() if p.is_file() and _is_image(p)])

    assert audio_paths, "No audio files found in files/"
    assert image_paths, "No image files found in files/"

    # Use the smallest count to build pairs; require at least one pair
    num_pairs = min(len(audio_paths), len(image_paths))
    assert num_pairs >= 1, "Need at least one audio and one image"

    selected_audios = audio_paths[:num_pairs]
    selected_images = image_paths[:num_pairs]

    # Build multipart body: first half audio, second half images
    files = []
    for ap in selected_audios:
        mime = mimetypes.guess_type(str(ap))[0] or "application/octet-stream"
        files.append(("files", (ap.name, ap.open("rb"), mime)))
    for ip in selected_images:
        mime = mimetypes.guess_type(str(ip))[0] or "application/octet-stream"
        files.append(("files", (ip.name, ip.open("rb"), mime)))

    if OUTPUT_VIDEO.exists():
        OUTPUT_VIDEO.unlink()

    client = TestClient(app)
    response = client.post("/stitch", files=files)

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("video/mp4")

    # Save the returned MP4
    OUTPUT_VIDEO.write_bytes(response.content)

    # Basic sanity checks on the saved file
    assert OUTPUT_VIDEO.exists()
    assert OUTPUT_VIDEO.stat().st_size > 0 