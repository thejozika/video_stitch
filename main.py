from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import tempfile
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, VideoClip
from typing import Optional
import base64
import re
import httpx
from urllib.parse import urlparse
import numpy as np

app = FastAPI()


def _create_reveal_wipe_segment(image_path: str, audio_clip: AudioFileClip, total_duration: float) -> VideoClip:
    """
    Creates an ImageClip with a time-varying mask:
    - Reveal: top-to-bottom over in_duration
    - Hold fully visible
    - Wipe out: left-to-right over out_duration (clears like a chalkboard)
    """
    in_duration = min(0.8, max(0.2, total_duration * 0.2))
    out_duration = min(0.6, max(0.2, total_duration * 0.15))
    image_clip = ImageClip(image_path)
    width, height = image_clip.size

    def mask_make_frame(t: float):
        # Values in [0, 1]
        if t <= in_duration:
            # Reveal from top to bottom
            progress = np.clip(t / max(in_duration, 1e-6), 0.0, 1.0)
            y_threshold = int(progress * height)
            frame = np.zeros((height, width), dtype=float)
            frame[:y_threshold, :] = 1.0
            return frame
        elif t >= total_duration - out_duration:
            # Wipe out from left to right (visible region shrinks to the right)
            progress = np.clip((t - (total_duration - out_duration)) / max(out_duration, 1e-6), 0.0, 1.0)
            x_threshold = int(progress * width)
            frame = np.ones((height, width), dtype=float)
            frame[:, :x_threshold] = 0.0
            return frame
        else:
            return np.ones((height, width), dtype=float)

    mask_clip = VideoClip(make_frame=mask_make_frame, ismask=True).set_duration(total_duration)
    mask_clip = mask_clip.set_fps(24)

    return image_clip.set_duration(total_duration).set_audio(audio_clip).set_mask(mask_clip)


def _stitch_from_temp_paths(temp_paths: List[str], num_pairs: int, temp_dir: str) -> str:
    output_path = os.path.join(temp_dir, "output.mp4")

    clips: List = []
    # Build clips: pair i-th audio with i-th image
    for i in range(num_pairs):
        audio_path = temp_paths[i]
        image_path = temp_paths[num_pairs + i]

        try:
            audio_clip = AudioFileClip(audio_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read audio file #{i+1}: {exc}")

        try:
            # Build an animated segment with reveal + wipe
            total_duration = (audio_clip.duration or 0) + 2.0
            if total_duration <= 0:
                audio_clip.close()
                raise HTTPException(status_code=400, detail=f"Audio file #{i+1} has invalid duration.")
            segment = _create_reveal_wipe_segment(image_path, audio_clip, total_duration)
        except Exception as exc:
            audio_clip.close()
            raise HTTPException(status_code=400, detail=f"Failed to build image segment #{i+1}: {exc}")

        clips.append(segment)

    if not clips:
        raise HTTPException(status_code=400, detail="No valid media segments were created.")

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        temp_audiofile=os.path.join(temp_dir, "temp-audio.m4a"),
        remove_temp=True,
        verbose=False,
        logger=None,
    )

    final_clip.close()
    for c in clips:
        try:
            c.close()
        except Exception:
            pass

    return output_path


def _ext_from_mime(mime: Optional[str]) -> str:
    if not mime:
        return ".bin"
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/aac": ".aac",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/bmp": ".bmp",
        "image/webp": ".webp",
    }
    return mapping.get(mime.lower(), ".bin")


def _decode_base64_to_tempfile(data_str: str, temp_dir: str, suggested_name: Optional[str] = None) -> str:
    mime = None
    m = re.match(r"^data:([^;]+);base64,(.*)$", data_str, re.DOTALL)
    if m:
        mime = m.group(1)
        b64 = m.group(2)
    else:
        b64 = data_str

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {exc}")

    suffix = os.path.splitext(suggested_name)[1] if suggested_name else ""
    if not suffix:
        suffix = _ext_from_mime(mime)

    tmp = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=suffix or ".bin")
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name


def _download_url_to_tempfile(url: str, temp_dir: str, suggested_name: Optional[str] = None) -> str:
    parsed = urlparse(url)
    url_suffix = os.path.splitext(parsed.path)[1]

    timeout = httpx.Timeout(30.0, connect=10.0)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type")
            suffix = os.path.splitext(suggested_name)[1] if suggested_name else ""
            if not suffix:
                suffix = url_suffix
            if not suffix and content_type:
                suffix = _ext_from_mime(content_type.split(";")[0].strip())
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=suffix or ".bin")
            tmp.write(resp.content)
            tmp.flush()
            tmp.close()
            return tmp.name
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {exc}")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/stitch")
async def stitch_media(files: List[UploadFile] = File(...)):
    """
    Accepts an array of uploaded files where the first half are audio files and the second half are image files.
    For each pair (audio_i, image_i), create a video segment where the image is displayed for (audio duration + 2) seconds
    with the audio overlaid. Concatenate all segments and return the final MP4.
    """
    if not files or len(files) < 2 or len(files) % 2 != 0:
        raise HTTPException(status_code=400, detail="Expected an even number of files: first half audio, second half images.")

    num_pairs = len(files) // 2

    # Create a temporary workspace for processing
    temp_dir = tempfile.mkdtemp(prefix="stitch_")

    temp_paths: List[str] = []
    try:
        # Persist uploads to disk
        for upload in files:
            filename = upload.filename or "upload"
            suffix = os.path.splitext(filename)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=suffix)
            content = await upload.read()
            tmp.write(content)
            tmp.flush()
            tmp.close()
            temp_paths.append(tmp.name)

        output_path = _stitch_from_temp_paths(temp_paths, num_pairs, temp_dir)
        return FileResponse(path=output_path, filename="stitched.mp4", media_type="video/mp4")
    finally:
        pass


from pydantic import BaseModel
from typing import Optional


class DataItem(BaseModel):
    data: Optional[str] = None
    filename: Optional[str] = None
    img_url: Optional[str] = None
    url: Optional[str] = None


class StitchBase64Request(BaseModel):
    files: Optional[List[str]] = None
    filenames: Optional[List[Optional[str]]] = None
    audio: Optional[List[DataItem]] = None
    image: Optional[List[DataItem]] = None
    images: Optional[List[DataItem]] = None


@app.post("/stitch_base64")
async def stitch_media_base64(payload: StitchBase64Request):
    temp_dir = tempfile.mkdtemp(prefix="stitch_b64_")

    temp_paths: List[str] = []
    try:
        # Legacy shape: flat files array (first half audio, second half images)
        if payload.files is not None:
            if len(payload.files) < 2 or len(payload.files) % 2 != 0:
                raise HTTPException(status_code=400, detail="Expected an even number of base64 files: first half audio, second half images.")

            num_pairs = len(payload.files) // 2
            file_names = payload.filenames or [None] * len(payload.files)
            if len(file_names) != len(payload.files):
                raise HTTPException(status_code=400, detail="filenames length must match files length when provided.")

            for data_str, name in zip(payload.files, file_names):
                path = _decode_base64_to_tempfile(data_str, temp_dir, name)
                temp_paths.append(path)

            output_path = _stitch_from_temp_paths(temp_paths, num_pairs, temp_dir)
            return FileResponse(path=output_path, filename="stitched.mp4", media_type="video/mp4")

        # New shape: audio base64 + image URLs
        if payload.audio is not None and (payload.image is not None or payload.images is not None):
            image_items = payload.image if payload.image is not None else payload.images
            num_pairs = min(len(payload.audio), len(image_items))
            if num_pairs == 0:
                raise HTTPException(status_code=400, detail="Need at least one audio and one image")

            audio_paths: List[str] = []
            for item in payload.audio[:num_pairs]:
                if not item.data:
                    raise HTTPException(status_code=400, detail="Each audio item must include base64 in 'data'")
                audio_paths.append(_decode_base64_to_tempfile(item.data, temp_dir, item.filename))

            image_paths: List[str] = []
            for item in image_items[:num_pairs]:
                chosen_url = item.img_url or item.url
                if chosen_url:
                    image_paths.append(_download_url_to_tempfile(chosen_url, temp_dir, item.filename))
                elif item.data:
                    image_paths.append(_decode_base64_to_tempfile(item.data, temp_dir, item.filename))
                else:
                    raise HTTPException(status_code=400, detail="Each image item must include 'img_url'/'url' or base64 'data'")

            temp_paths = audio_paths + image_paths
            output_path = _stitch_from_temp_paths(temp_paths, num_pairs, temp_dir)
            return FileResponse(path=output_path, filename="stitched.mp4", media_type="video/mp4")

        raise HTTPException(status_code=400, detail="Provide either 'files' or both 'audio' and 'image(s)'")
    finally:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)