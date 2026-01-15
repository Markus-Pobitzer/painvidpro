import os
from typing import Tuple

import ffmpeg


def convert_video_with_ffmpeg(video_path: str, temp_path: str = "temp_fixed.mp4") -> Tuple[bool, str]:
    """
    Converts a video (specifically to fix AV1 codec issues) to H.264
    and overwrites the original file.

    Args:
        file_path (str): The path to the .mp4 file to be converted.
        temp_path (str): File to store it temporally.

    Returns:
        success (bool), msg (str): If not success the msg indicates the error.
    """
    try:
        # 1. Build the conversion pipeline
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream,
            temp_path,
            vcodec="libx264",
            pix_fmt="yuv420p",  # Pixel color format compatible with cv2
            crf=23,
            preset="medium",
            an=None,  # Strip audio
        )

        # 2. Run the process (.run() triggers the actual work)
        # overwrite_output=True allows FFmpeg to overwrite temp_path if it exists
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

        # 3. Swap files
        os.remove(video_path)
        os.rename(temp_path, video_path)

    except ffmpeg.Error as e:
        # ffmpeg-python errors contain the full stderr from the terminal
        return False, f"FFmpeg error: {e.stderr.decode()}"
    except Exception as e:
        return False, str(e)

    return True, ""
