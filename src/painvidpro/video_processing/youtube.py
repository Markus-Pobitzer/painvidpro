"""File to download and process videos from YouTube."""

import yt_dlp


def download_video(url: str, output_path: str):
    """Downloads a YouTube video.

    Args:
        url: The YouTube URL.
        output_path: The output path on disk.
    """
    ydl_opts = {"format": "bestvideo[height<=360]", "outtmpl": output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
