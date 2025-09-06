from fastapi import FastAPI
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import tempfile
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

app = FastAPI()


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
    audio_files = files[:num_pairs]
    image_files = files[num_pairs:]

    # Create a temporary workspace for processing
    temp_dir = tempfile.mkdtemp(prefix="stitch_")
    output_path = os.path.join(temp_dir, "output.mp4")

    temp_paths = []
    clips = []
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
        # Write the video. Set fps for static images to ensure proper encoding.
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

        # Close all clips to release resources
        final_clip.close()
        for c in clips:
            try:
                # ImageClip(AudioFileClip) closes when parent closed; still call close safely
                c.close()
            except Exception:
                pass

        return FileResponse(path=output_path, filename="stitched.mp4", media_type="video/mp4")
    finally:
        # MoviePy needs the files to exist while writing. We clean the temp directory on process exit or by caller.
        # Do not remove temp_dir here before returning the FileResponse streaming completes.
        # Leaving files for the OS to clean (tmp) is acceptable for short-lived processes; otherwise, implement background cleanup.
        pass
