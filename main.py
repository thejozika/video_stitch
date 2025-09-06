from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import tempfile
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
from typing import Optional
import base64
import re

app = FastAPI()


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
            image_clip = ImageClip(image_path)
        except Exception as exc:
            audio_clip.close()
            raise HTTPException(status_code=400, detail=f"Failed to read image file #{i+1}: {exc}")

        duration_with_tail = (audio_clip.duration or 0) + 2.0
        if duration_with_tail <= 0:
            image_clip.close()
            audio_clip.close()
            raise HTTPException(status_code=400, detail=f"Audio file #{i+1} has invalid duration.")

        segment = image_clip.set_duration(duration_with_tail).set_audio(audio_clip)
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


class StitchBase64Request(BaseModel):
    files: List[str]
    filenames: Optional[List[Optional[str]]] = None


@app.post("/stitch_base64")
async def stitch_media_base64(payload: StitchBase64Request):
    if not payload.files or len(payload.files) < 2 or len(payload.files) % 2 != 0:
        raise HTTPException(status_code=400, detail="Expected an even number of base64 files: first half audio, second half images.")

    num_pairs = len(payload.files) // 2
    temp_dir = tempfile.mkdtemp(prefix="stitch_b64_")

    temp_paths: List[str] = []
    try:
        file_names = payload.filenames or [None] * len(payload.files)
        if len(file_names) != len(payload.files):
            raise HTTPException(status_code=400, detail="filenames length must match files length when provided.")

        for data_str, name in zip(payload.files, file_names):
            path = _decode_base64_to_tempfile(data_str, temp_dir, name)
            temp_paths.append(path)

        output_path = _stitch_from_temp_paths(temp_paths, num_pairs, temp_dir)
        return FileResponse(path=output_path, filename="stitched.mp4", media_type="video/mp4")
    finally:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)